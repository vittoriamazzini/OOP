import re
import fnmatch
import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal, stats
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages


# multiple file analyzer
"""Quando invece analizzi la cartella, che contiene 3 sottocartelle, devi aprire tutti 
i file, fare le analisi come se fossero file singoli e poi fare gli istogrammi di v_bd 
e R_q, nonché confrontare le due misure ad LN2 e le due misure di Aprile. Per la 
directory, trova il modo di tenere separate le tre sottocartelle in results (vedi come 
ho fatto io nel codice, tanto è lìunico modo per farlo), in modo da poi poter fare 
facilmente gli istogrammi a LN2 o quelli di Aprile. Sempre per la directory, come nel 
caso claro, fatti la lista di file e poi chiama per ciascuno una istanza della classe 
Single, così il codice lo scrivi solo una volta"""


class SingleFile:
    def __init__(self, path):
        self.path = path
        self.fileinfo = _get_fileinfo(path)
        self.df_grouped = {}

    def file_reader(self):
        path = self.path

        def header_reader(self):
            with open(path, "r") as file:
                for number, line in enumerate(file):
                    if "SiPM" in line:
                        return number
                    else:
                        print("Error : header not found")
                        sys.exit(1)

        self.df = pd.read_csv(path, header=header_reader(path))
        self.df_sorted = self.df.sort_values(by=["SiPM", "Step"], ignore_index=True)
        self.df_grouped = self.df_sorted.groupby("SiPM")

        return self.df_grouped

    def single_analyzer(self, savepath=os.getcwd()):
        linear_roomT = 0.75
        linear_LN2 = 1.55
        peak_width = 15

        # Create the savepath folder if it doesn't exist
        if not os.path.exists(savepath):
            os.makedirs(savepath)

        if self.fileinfo["temp"] == "LN2":
            start_fit = linear_LN2
        else:
            start_fit = linear_roomT

        if self.fileinfo["direction"] == "f":
            analyzer_func = forward_analyzer
            result_cols = ["SiPM", "R_quenching", "R_quenching_std"]
            result_file_suffix = "Forward_results"
            plot_func = forward_plotter
            plot_file_suffix = "Forward"
            parameter = start_fit
        elif self.fileinfo["direction"] == "r":
            analyzer_func = reverse_analyzer
            result_cols = ["SiPM", "V_bd", "V_bd_std"]
            result_file_suffix = "Reverse_results"
            plot_func = reverse_plotter
            plot_file_suffix = "Reverse"
            parameter = peak_width
        else:
            print(f"Error: incorrect polarization value (give a value as 'f' or 'r')")

        resulting_df = self.df_grouped.apply(analyzer_func, parameter)
        joined_df = self.df_sorted.join(resulting_df, on="SiPM")

        output_df = joined_df[result_cols].drop_duplicates(subset="SiPM")
        result_file_name = f"Arduino{self.fileinfo['ardu']}_Test{self.fileinfo['test']}_Temp{self.fileinfo['temp']}_{result_file_suffix}.csv"
        output_df.to_csv(os.path.join(savepath, result_file_name), index=False)

        plot_file_name = f"Arduino{self.fileinfo['ardu']}_Test{self.fileinfo['test']}_Temp{self.fileinfo['temp']}_{plot_file_suffix}.pdf"
        pdf_pages = PdfPages(os.path.join(savepath, plot_file_name))
        joined_df.groupby("SiPM").apply(plot_func, pdf_pages)
        pdf_pages.close()


class MultipleFiles:
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.matching_files = []

    def read_folder(self):
        for root, dirs, filenames in tqdm(os.walk(self.root_folder), "Reading files"):
            for filename in filenames:
                if fnmatch.fnmatch(os.path.join(root, filename), "*.csv"):
                    self.matching_files.append(os.path.join(root, filename))
        return self.matching_files

    def dir_analyzer(self):
        for file in tqdm(self.matching_files, "Analyzing files"):
            try:
                subfolder = re.search(".+\\\\(.+?)\\\\ARDU_.+", file).group(1)
            except AttributeError:
                subfolder = ""

            single = SingleFile(file)
            single.file_reader()
            single.single_analyzer(
                savepath=os.path.join(os.getcwd(), "resulting analysis", subfolder)
            )

            # Introduced to solve memory issues when dealing with big folders
            matplotlib.use("agg")

    def create_histogram(self, compare_day=True, compare_temp=True):
        def df_join(directory, direction):
            files = [
                file
                for file in os.listdir(directory)
                if direction in file and file.endswith(".csv")
            ]
            dfs = [pd.read_csv(os.path.join(directory, file)) for file in files]
            data = pd.concat(dfs)
            data["subdir"] = os.path.basename(directory)
            return data

        resulting_values = os.path.join(os.getcwd(), "results")
        fwd_all = []
        rev_all = []

        for subdir, dirs, files in os.walk(resulting_values):
            for dir in dirs:
                subdir_path = os.path.join(subdir, dir)

                # Retrieve all the data of fwd and rev
                forward_data = df_join(subdir_path, "Forward")
                reverse_data = df_join(subdir_path, "Reverse")

                if (
                    compare_day == True or compare_temp == True
                ):  # Create the full df only if needed
                    fwd_all.append(forward_data)
                    rev_all.append(reverse_data)

                # Plot R_q and V_bd hist for each subfolder
                fig, axs = plt.subplots(2)
                hist_params(fig, axs, dir)
                forward_data.plot.hist(
                    column=["R_quenching"],
                    ax=axs[0],
                    bins=15,
                    range=(
                        min(forward_data["R_quenching"]),
                        max(forward_data["R_quenching"]),
                    ),
                    color="darkgreen",
                    alpha=0.7,
                )
                reverse_data.plot.hist(
                    column=["V_bd"],
                    ax=axs[1],
                    bins=15,
                    range=(min(reverse_data["V_bd"]), max(reverse_data["V_bd"])),
                    color="darkorange",
                    alpha=0.7,
                )
                plt.tight_layout()  # Prevents titles and axes from overlapping
                plotname = f"Histograms_{dir}.png"
                plt.savefig(
                    os.path.join(resulting_values, plotname), bbox_inches="tight"
                )
                plt.close()
                print(f"Plot saved as {resulting_values}\{plotname}")

        if compare_temp == True or compare_day == True:
            # Merge all the fwd and rev dataframes
            fwd_all = pd.concat(fwd_all)
            rev_all = pd.concat(rev_all)

        if compare_temp == True:
            ln2_fwd = fwd_all[fwd_all["subdir"].str.contains("LN2")]
            ln2_rev = rev_all[fwd_all["subdir"].str.contains("LN2")]

            fig, axs = plt.subplots(2)
            hist_params(fig, axs, "Liquid Nitrogen comparison")
            for subdir, group in ln2_fwd.groupby("subdir"):
                group["R_quenching"].hist(ax=axs[0], label=subdir, bins=15, alpha=0.6)
            for subdir, group in ln2_rev.groupby("subdir"):
                group["V_bd"].hist(ax=axs[1], label=subdir, bins=15, alpha=0.6)
            [ax.legend() for ax in axs]
            plt.tight_layout()
            plotname = f"LN2_comparison_hist.png"
            plt.savefig(os.path.join(resulting_values, plotname), bbox_inches="tight")
            plt.close()
            print(f"Plot saved as {resulting_values}\{plotname}")

        if compare_day == True:
            fwd_april = fwd_all[fwd_all["subdir"].str.contains("_04_")]
            rev_april = rev_all[fwd_all["subdir"].str.contains("_04_")]

            fig, axs = plt.subplots(2)
            hist_params(fig, axs, "April data comparison")
            for subdir, group in fwd_april.groupby("subdir"):
                group["R_quenching"].hist(ax=axs[0], label=subdir, bins=15, alpha=0.6)
            for subdir, group in rev_april.groupby("subdir"):
                group["V_bd"].hist(ax=axs[1], label=subdir, bins=15, alpha=0.6)
            [ax.legend() for ax in axs]
            plt.tight_layout()
            plotname = f"April_data_comparison_hist.png"
            plt.savefig(os.path.join(resulting_values, plotname), bbox_inches="tight")
            plt.close()
            print(f"Plot saved as {resulting_values}\{plotname}")


#################################


@staticmethod
def _get_fileinfo(path):
    """
    Extracts relevant information from a file path using regular expressions.

    Args:
        path (str): The file path to extract information from.

    Returns:
        dict: A dictionary containing the extracted information.
    """
    _ardu = re.search(".+ARDU_(.+?)_.+", path).group(1)
    _direction = re.search(".+[0-9]_(.+?)_.+", path).group(1)
    _test = re.search(".+Test_(.+?)_.+", path).group(1)
    _temp = re.search(".+_(.+?)_dataframe.+", path).group(1)

    _fileinfo = {"direction": _direction, "ardu": _ardu, "test": _test, "temp": _temp}

    return _fileinfo


def forward_analyzer(data_file, start_fit):
    # Convert data_file columns to numpy arrays
    x = np.array(data_file["V"])
    y = np.array(data_file["I"])

    # Select the data points that should be used for the linear fit
    fit_x = x[x >= start_fit]
    fit_y = y[x >= start_fit]

    # Perform a linear regression on the selected data points
    slope, intercept, __, __, std_err = stats.linregress(fit_x, fit_y)

    # Calculate the quenching resistance and its standard deviation
    quenching_resistance = 1000 / slope
    quenching_res_std = max(std_err, 0.03 * quenching_resistance)

    # Store the results in a dictionary
    results = {
        "R_quenching": quenching_resistance,
        "R_quenching_std": quenching_res_std,
        "start_fit": start_fit,
        "slope": slope,
        "intercept": intercept,
    }

    # Return the results as a pandas series
    return pd.Series(results)


def forward_plotter(data_file, pdf):
    # Add a new column to the data_file DataFrame with linear values
    data_file["y_lin"] = data_file["slope"] * data_file["V"] + data_file["intercept"]

    # Get only the data for the linear fit
    lin_data = data_file[data_file["V"] >= data_file["start_fit"]]
    lin_x = lin_data["V"]
    lin_y = lin_data["y_lin"]

    # Create a figure and axis object
    fig, ax = plt.subplots()

    # Get the unique SiPM number
    sipm_number = lin_data["SiPM"].drop_duplicates().iloc[0]

    # Set the title of the figure
    fig.suptitle(f"Forward IV curve: SiPM {sipm_number}")

    # Plot the linear fit and data
    ax.plot(
        lin_x,
        lin_y,
        color="coral",
        linewidth=1.2,
        label=f'Linear fit: Rq = ({lin_data["R_quenching"].iloc[0]:.2f} $\pm$ {lin_data["R_quenching_std"].iloc[0]:.2f}) $\Omega$',
        zorder=2,
    )
    ax.errorbar(
        data_file["V"], data_file["I"], data_file["I_err"], marker=".", zorder=1
    )

    # Label the x and y axis
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (mA)")
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc="upper left")

    pdf.savefig()
    plt.close()


def reverse_analyzer(data_file, peak_width):
    x = np.array(data_file["V"])
    y = np.array(data_file["I"])

    def norm_derivative(x, y):
        dy_dx = np.gradient(y) / np.gradient(x)
        return 1 / y * dy_dx

    # Evaluation of the 1st derivative
    derivative = norm_derivative(x, y)

    # 5th degree polynomial fit
    coefs = np.polyfit(x, derivative, 5)
    y_fit = np.polyval(coefs, x)

    # Peak finder
    peaks, _ = signal.find_peaks(y_fit, width=peak_width)
    if len(peaks) > 0:
        idx_max = peaks[0]
        x_max = x[idx_max]
        fwhm = x[int(idx_max + peak_width / 2)] - x[int(idx_max - peak_width / 2)]

        # Second degree polynomial fit around the peak
        x_poly = x[np.logical_and(x >= (x_max - fwhm), x <= (x_max + fwhm))]
        y_poly = y_fit[np.logical_and(x >= (x_max - fwhm), x <= (x_max + fwhm))]
        poly_coefs_peak = np.polyfit(x_poly, y_poly, 2)

    else:
        poly_coefs_peak = [np.nan, np.nan, np.nan]
        fwhm = np.nan

    # Returning the values
    results = {
        "V_bd": x_max,
        "V_bd_std": fwhm / 2,
        "width": fwhm,
        "coefs": coefs,
        "poly_coefs_peak": poly_coefs_peak,
    }
    # Return the results as a pandas series
    return pd.Series(results)


def reverse_plotter(data_file, pdf):
    x = np.array(data_file["V"])
    y = np.array(data_file["I"])

    V_bd = data_file["V_bd"].iloc[0]
    coefs = data_file["coefs"].iloc[0]  # fifth degree coefs

    def norm_derivative(x, y):
        dy_dx = np.gradient(y) / np.gradient(x)
        return 1 / y * dy_dx

    derivative = norm_derivative(x, y)

    y_poly_fifth_deg = (
        coefs[5]
        + coefs[4] * x
        + coefs[3] * x**2
        + coefs[2] * x**3
        + coefs[1] * x**4
        + coefs[0] * x**5
    )
    x_poly_second_deg = x[
        np.logical_and(
            x >= (V_bd - data_file["width"].iloc[0] / 2),
            x <= (V_bd + data_file["width"].iloc[0] / 2),
        )
    ]
    poly_coefs_peak = data_file["poly_coefs_peak"].iloc[0]
    y_poly_peak = (
        poly_coefs_peak[2]
        + poly_coefs_peak[1] * x_poly_second_deg
        + poly_coefs_peak[0] * x_poly_second_deg**2
    )

    sipm_number = list(data_file["SiPM"].drop_duplicates())[0]

    fig, ax = plt.subplots()
    fig.suptitle(f"Reverse IV curve: SiPM {sipm_number}")
    ax.set_yscale("log")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (mA)")
    ax2 = ax.twinx()
    ax2.set_ylabel(r"$I^{-1} \frac{dI}{dV}$", color="black")
    ax2.tick_params(axis="y", colors="black")

    ax.errorbar(
        data_file["V"],
        data_file["I"],
        data_file["I_err"],
        marker=".",
        color="cadetblue",
        label="Data",
    )
    ax.grid(True)

    ax2.scatter(x, derivative, marker="o", s=5, color="coral", label="Derivative")
    ax2.plot(x, y_poly_fifth_deg, color="limegreen", label="5th-deg polynomial")
    ax2.plot(
        x_poly_second_deg,
        y_poly_peak,
        color="red",
        label="Second degree around peak",
    )
    ax2.axvline(
        V_bd,
        color="gold",
        label=f"V_bd = {V_bd:.2f} ± {abs(data_file['V_bd_std'].iloc[0]):.2f} V",
    )

    # Add legends and adjust layout
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")
    plt.subplots_adjust(hspace=0.05)

    pdf.savefig()
    plt.close()


def hist_params(fig, axs, dir):
    """
    This function takes as input a figure and two axes (as a list) and sets the titles, labels and grids for the histograms.
    The first histogram is for the quenching resistance and the second one is for the breakdown voltage.
    Args:
        fig (matplotlib.figure): Figure to set the title of
        axs (list): lsit of the axes of the plot
        dir (str): direction of the IV curve
    """
    fig.suptitle(f"{dir}: R_q and V_Bd distribution")
    [ax.grid("on") for ax in axs]
    axs[0].set_title("Quenching Resistance Histogram")
    axs[0].set_xlabel("$R_q [\Omega$]")
    axs[0].set_ylabel("Frequency")
    axs[1].set_title("Breakdown Voltage Histogram")
    axs[1].set_xlabel("$V_{Bd}$ [V]")
    axs[1].set_ylabel("Frequency")
