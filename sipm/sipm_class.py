import re
import fnmatch
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import signal, stats
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages


class SingleFile:
    """
    A class for reading and analyzing a single data file.
    """

    def __init__(self, path):
        """
        Initializes a SingleFile object.

        Parameters:
        path (str): The path to the data file to be analyzed.
        """
        self.path = path
        self.fileinfo = _get_fileinfo(path)
        self.df_grouped = {}

    def file_reader(self):
        """
        Reads the data file and groups it by SiPM.

        Returns: a grouped dataframe.
        """
        path = self.path

        def header_reader(self):
            """
            Helper function to determine the header row of the data file.

            Parameters:
            self: A SingleFile object.

            Returns:
            int: The row number of the header.
            """
            with open(path, "r") as file:
                for number, line in enumerate(file):
                    if "SiPM" in line:
                        return number
                    else:
                        print("Error : header not found")
                        exit()

        self.df = pd.read_csv(path, header=header_reader(path))
        self.df_sorted = self.df.sort_values(by=["SiPM", "Step"], ignore_index=True)
        self.df_grouped = self.df_sorted.groupby("SiPM")

        return self.df_grouped

    def single_analyzer(self, savepath=os.getcwd()):
        """
        Analyzes the data and generates results and plots.

        Parameters:
        savepath (str): The directory where the results and plots will be saved.

        Returns: None
        """

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
            analyzer_function = forward_analyzer
            resulting_columns = ["SiPM", "R_quenching", "R_quenching_std"]
            resulting_suffix = "Forward_results"
            plotter_function = forward_plotter
            plot_suffix = "Forward"
            parameter = start_fit
        elif self.fileinfo["direction"] == "r":
            analyzer_function = reverse_analyzer
            resulting_columns = ["SiPM", "V_bd", "V_bd_std"]
            resulting_suffix = "Reverse_results"
            plotter_function = reverse_plotter
            plot_suffix = "Reverse"
            parameter = peak_width
        else:
            print(f"Error: incorrect polarization value (give a value as 'f' or 'r')")

        resulting_df = self.df_grouped.apply(analyzer_function, parameter)
        joined_df = self.df_sorted.join(resulting_df, on="SiPM")

        output_df = joined_df[resulting_columns].drop_duplicates(subset="SiPM")
        resulting_filename = f"Arduino{self.fileinfo['ardu']}_Test{self.fileinfo['test']}_Temp{self.fileinfo['temp']}_{resulting_suffix}.csv"
        output_df.to_csv(os.path.join(savepath, resulting_filename), index=False)

        plot_file_name = f"Arduino{self.fileinfo['ardu']}_Test{self.fileinfo['test']}_Temp{self.fileinfo['temp']}_{plot_suffix}.pdf"
        pdf_pages = PdfPages(os.path.join(savepath, plot_file_name))
        joined_df.groupby("SiPM").apply(plotter_function, pdf_pages)
        pdf_pages.close()


class MultipleFiles:
    """
    A class to analyze multiple CSV files.
    """

    def __init__(self, root_folder):
        """
        Initializes MultipleFiles with the root folder path and an empty list of matching files.
        """
        self.root_folder = root_folder
        self.matching_files = []

    def read_folder(self):
        """
        Searches for CSV files within the root folder and its subdirectories.

        Returns:
        matching_files (list): a list of CSV files within the root folder.
        """
        for root, dirs, filenames in os.walk(self.root_folder):
            for filename in filenames:
                if fnmatch.fnmatch(os.path.join(root, filename), "*.csv"):
                    self.matching_files.append(os.path.join(root, filename))
        return self.matching_files

    def dir_analyzer(self):
        """
        Analyzes the matching CSV files and saves the results.

        The method uses the SingleFile class to read and analyze the data in each CSV file.
        It saves the results in a folder named "resulting sipm analys", with a subfolder for each
        group of CSV files.
        """
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

    def create_histogram(self):
        """
        Creates and saves histograms of the analyzed data.

        The method uses the resulting CSV files saved in the "resulting sipm analys" folder to
        create histograms for each group of files and for different conditions.
        """

        def df_join(results_directory, data_type):
            """
            Joins the CSV files in a given folder into a single pandas DataFrame.

            Args:
            results_directory (str): the path to the folder containing the CSV files.
            data_type (str): a string to filter the CSV files to be joined.

            Returns:
            data (DataFrame): a pandas DataFrame containing the joined data.
            """
            files = [
                file
                for file in os.listdir(results_directory)
                if data_type in file and file.endswith(".csv")
            ]
            dfs = [pd.read_csv(os.path.join(results_directory, file)) for file in files]
            data = pd.concat(dfs)
            data["subdir"] = os.path.basename(results_directory)
            return data

        resulting_values = os.path.join(os.getcwd(), "resulting analysis")

        forward_files = []
        reverse_files = []

        for subdir, dirs, files in os.walk(resulting_values):
            for dir in dirs:
                subdir_path = os.path.join(subdir, dir)

                try:
                    forward_data = df_join(subdir_path, "Forward")
                    reverse_data = df_join(subdir_path, "Reverse")
                except Exception as e:
                    print(f"Error processing data in {subdir_path}: {e}")
                    continue

                forward_files.append(forward_data)
                reverse_files.append(reverse_data)

                fig, axs = plt.subplots(2)
                fig.suptitle(f"{dir}: R_q and V_Bd distribution")

                axs[0].set_title("Quenching Resistance Histogram")
                axs[0].set_xlabel("$R_q [\Omega$]")
                axs[0].set_ylabel("Frequency")
                axs[1].set_title("Breakdown Voltage Histogram")
                axs[1].set_xlabel("$V_{Bd}$ [V]")
                axs[1].set_ylabel("Frequency")

                forward_data.plot.hist(
                    column=["R_quenching"],
                    ax=axs[0],
                    bins=15,
                    range=(
                        min(forward_data["R_quenching"]),
                        max(forward_data["R_quenching"]),
                    ),
                    color="limegreen",
                    alpha=1,
                )

                reverse_data.plot.hist(
                    column=["V_bd"],
                    ax=axs[1],
                    bins=15,
                    range=(min(reverse_data["V_bd"]), max(reverse_data["V_bd"])),
                    color="cadetblue",
                    alpha=1,
                )

                plt.tight_layout()
                plotname = f"Histograms_{dir}.png"

                plt.savefig(
                    os.path.join(resulting_values, plotname), bbox_inches="tight"
                )
                plt.close()
                print(f"Plot saved as {os.path.join(resulting_values, plotname)}")

        all_files = pd.merge(
            pd.concat(forward_files), pd.concat(reverse_files), on=["SiPM", "subdir"]
        )

        def plot_comparison_hist(data, title, subdir_filter, plotname):
            """
            Helper function to plot histograms of the R_quenching and V_bd columns.
            Args:
                data (pd.DataFrame): A pandas DataFrame containing the data to be plotted.
                title (str): The title of the plot.
                subdir_filter (str): Filter for the data by subdirectory name.
                plotname (str): The name of the file to save the plot to.

            Returns: None
            """

            y_vals = ["R_quenching", "V_bd"]

            fig, axs = plt.subplots(2, figsize=(8, 8))
            fig.suptitle(f"{title}: R_q and V_Bd distribution")
            axs[0].set_xlabel("$R_q [\Omega$]")
            axs[0].set_ylabel("Frequency")
            axs[1].set_xlabel("$V_{Bd}$ [V]")
            axs[1].set_ylabel("Frequency")

            # Check the comparison condition
            matching_df = data[data["subdir"].str.contains(subdir_filter)]

            for i, y_val in enumerate(y_vals):
                for subdir, group in matching_df.groupby("subdir"):
                    group[y_val].hist(ax=axs[i], label=subdir, bins=15, alpha=0.6)

            [ax.legend() for ax in axs]

            plt.tight_layout()

            plt.savefig(os.path.join(resulting_values, plotname), bbox_inches="tight")
            plt.close()

            print(f"Plot saved as {os.path.join(resulting_values, plotname)}")

        plot_comparison_hist(
            all_files,
            "Liquid Nitrogen comparison",
            "LN2",
            "LN2_comparison_hist.png",
        )

        plot_comparison_hist(
            all_files,
            "April data comparison",
            "_04_",
            "April_data_comparison_hist.png",
        )


######################## Static Methods ##############################


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


@staticmethod
def forward_analyzer(data_file, start_fit):
    """
    Analyze the forward IV curve file and returns the results.

    Args:
        data_file (pandas.DataFrame): A DataFrame containing the reverse bias IV data.
        peak_width (int): The width of the peaks in the 5th degree polynomial fit.

    Returns:
        pandas.Series: A Series containing the breakdown voltage, breakdown voltage standard deviation, FWHM,
        the coefficients of the 5th degree polynomial fit, and the coefficients of the 2nd degree polynomial fit
        around the peak.

    """
    x = np.array(data_file["V"])
    y = np.array(data_file["I"])

    fit_x = x[x >= start_fit]
    fit_y = y[x >= start_fit]

    slope, intercept, __, __, std_err = stats.linregress(fit_x, fit_y)

    quenching_resistance = 1000 / slope
    quenching_res_std = max(std_err, 0.03 * quenching_resistance)

    results = {
        "R_quenching": quenching_resistance,
        "R_quenching_std": quenching_res_std,
        "start_fit": start_fit,
        "slope": slope,
        "intercept": intercept,
    }

    return pd.Series(results)


@staticmethod
def forward_plotter(data_file, pdf):
    """
    Plot the forward IV curve for a given SiPM and save the plot to a PDF.

    Parameters:
    -----------
    data_file : pandas DataFrame
        DataFrame with the data for the SiPM to be plotted.
    pdf : matplotlib.backends.backend_pdf.PdfPages
        PDF file to save the plot to.

    Returns: None
    """

    # Add a new column to the data_file DataFrame with linear values
    data_file["y_lin"] = data_file["slope"] * data_file["V"] + data_file["intercept"]

    # Get only the data for the linear fit
    lin_data = data_file[data_file["V"] >= data_file["start_fit"]]
    lin_x = lin_data["V"]
    lin_y = lin_data["y_lin"]

    fig, ax = plt.subplots()

    sipm_number = lin_data["SiPM"].drop_duplicates().iloc[0]

    fig.suptitle(f"Forward IV curve: SiPM {sipm_number}")
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

    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (mA)")
    lines, labels = ax.get_legend_handles_labels()
    ax.legend(lines, labels, loc="upper left")

    pdf.savefig()
    plt.close()


@staticmethod
def reverse_analyzer(data_file, peak_width):
    """Analyzes a reverse IV curve file and returns the results.

    Args:
        data_file (pandas.DataFrame): A DataFrame containing the reverse IV curve data.
            It should have columns "V" and "I" for the voltage and current data, respectively.
        peak_width (int): The expected width of the peak in the derivative curve, in number of data points.

    Returns:
        pandas.Series: A Series containing the analysis results. The following values are included:

        - "V_bd": The breakdown voltage of the SiPM, in volts.
        - "V_bd_std": The uncertainty in the breakdown voltage, in volts.
        - "width": The full width at half maximum (FWHM) of the peak in the derivative curve, in volts.
        - "coefs": A NumPy array containing the coefficients of the 5th degree polynomial fit to the derivative curve.
        - "poly_coefs_peak": A NumPy array containing the coefficients of the 2nd degree polynomial fit around the peak.
    """

    x = np.array(data_file["V"])
    y = np.array(data_file["I"])

    def norm_derivative(x, y):
        dy_dx = np.gradient(y) / np.gradient(x)
        return 1 / y * dy_dx

    derivative = norm_derivative(x, y)

    coefs = np.polyfit(x, derivative, 5)
    y_fit = np.polyval(coefs, x)

    peaks, _ = signal.find_peaks(y_fit, width=peak_width)
    if len(peaks) > 0:
        idx_max = peaks[0]
        x_max = x[idx_max]
        fwhm = x[int(idx_max + peak_width / 2)] - x[int(idx_max - peak_width / 2)]

        x_poly = x[np.logical_and(x >= (x_max - fwhm), x <= (x_max + fwhm))]
        y_poly = y_fit[np.logical_and(x >= (x_max - fwhm), x <= (x_max + fwhm))]
        poly_coefs_peak = np.polyfit(x_poly, y_poly, 2)

    else:
        poly_coefs_peak = [np.nan, np.nan, np.nan]
        fwhm = np.nan

    results = {
        "V_bd": x_max,
        "V_bd_std": fwhm / 2,
        "width": fwhm,
        "coefs": coefs,
        "poly_coefs_peak": poly_coefs_peak,
    }

    return pd.Series(results)


@staticmethod
def reverse_plotter(data_file, pdf):
    """
    Plots the reverse current-voltage (IV) curve and its derivative, along with a 5th-degree polynomial fit
    and a second-degree polynomial fit around the peak of the IV curve.

    Parameters:
    -----------
    data_file : pandas DataFrame
        A DataFrame containing the IV curve data, with columns 'V', 'I', and 'I_err', as well as additional
        columns 'V_bd', 'V_bd_std', 'width', 'coefs', 'poly_coefs_peak', and 'SiPM'.
    pdf : matplotlib.backends.backend_pdf.PdfPages
        A PdfPages object to which the plot will be saved.

    Returns: None
    """
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

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")
    plt.subplots_adjust(hspace=0.05)

    pdf.savefig()
    plt.close()
