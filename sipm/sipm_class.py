import fnmatch
import re
import os
import sys
from tqdm import tqdm
import numpy as np
from numpy.polynomial import Polynomial
import pandas as pd
from scipy import stats, optimize, signal
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# single file forward
"""Per i file forward, bisogna leggere i dati e, per ciascun SiPM, eseguire un 
fit lineare sulla zona lineare dei dati (che inizia a 0.75 a roomT e a 1.55 per LN2). 
il valore 1000/m (con m coeff. angolare ) rappresenta la resistenza di quenching. 
Poi vanno plottati i dati ed il fit, salvando i risultati in un pdf che ha un plot 
di sipm diverso per pagina, quindi 30 pagine (fa tutto il pacchetto pdfpages). 
Infine vanno salvati i valori in un csv in maniera simile al claro """
# single file reverse
"""Per i file reverse si leggono i dati e per ciascun sipm si calcola la derivata 
normalizzata "1/i * di/dv" (con np.diff o np.gradient ad esempio), poi si fitta un 
polinomio di grado 5 su tale derivata e, sul massimo di questo polinomio, una gaussiana 
(o una parabola), il cui valore medio rappresenta la V di breakdown. Vanno plottati i 
dati, la derivata, il polinomio, la gaussiana ed il  valore di v_bd su un singolo plot 
che ha scala semilogy, anche qui salvando tutto in un pdf con 30 grafici, uno per sipm. 
Anche qui vanno salvati i dati in un csv
"""
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

    def single_analyzer(self):
        savepath = os.getcwd()
        linear_roomT = 0.75
        linear_LN2 = 1.55
        peak_width = 10

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
        elif self.fileinfo["direction"] == "r":
            analyzer_func = reverse_analyzer
            result_cols = ["SiPM", "V_bd", "V_bd_std"]
            result_file_suffix = "Reverse_results"
            plot_func = reverse_plotter
            plot_file_suffix = "Reverse"
        else:
            print(f"Error: incorrect polarization value (give a value as 'f' or 'r')")

        result_df = self.df_grouped.apply(analyzer_func, start_fit)
        joined_df = self.df_sorted.join(result_df, on="SiPM")

        out_df = joined_df[result_cols].drop_duplicates(subset="SiPM")
        result_file_name = f"Arduino{self.fileinfo['ardu']}_Test{self.fileinfo['test']}_Temp{self.fileinfo['temp']}_{result_file_suffix}.csv"
        out_df.to_csv(os.path.join(savepath, result_file_name), index=False)

        plot_file_name = f"Arduino{self.fileinfo['ardu']}_Test{self.fileinfo['test']}_Temp{self.fileinfo['temp']}_{plot_file_suffix}.pdf"
        pdf_pages = PdfPages(os.path.join(savepath, plot_file_name))
        joined_df.groupby("SiPM").apply(plot_func, pdf_pages)
        pdf_pages.close()


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
    fit_y = y[y >= start_fit]

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
        color="darkgreen",
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

    pdf.savefig()
    plt.close()


def reverse_analyzer(data, peak_width):
    x = data["V"].to_numpy()
    y = data["I"].to_numpy()

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

        # Gaussian fit around the peak
        x_gauss = x[np.logical_and(x >= (x_max - fwhm / 2), x <= (x_max + fwhm / 2))]
        y_gauss = y_fit[
            np.logical_and(x >= (x_max - fwhm / 2), x <= (x_max + fwhm / 2))
        ]
        fit_guess = [0, 1, x_max, fwhm / 2]

        def gauss(x, H, A, mu, sigma):
            return H + A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

        params, covar = optimize.curve_fit(
            gauss, x_gauss, y_gauss, fit_guess, maxfev=20000
        )

    else:
        params = [np.nan, np.nan, np.nan, np.nan]
        fwhm = np.nan

    # Returning the values
    results = {
        "V_bd": params[2],
        "V_bd_std": params[3],
        "width": fwhm,
        "coefs": coefs,
        "params": params,
    }
    # Return the results as a pandas series
    return pd.Series(results)


def reverse_plotter(data_file, pdf):
    x = np.array(data_file["V"])
    y = np.array(data_file["I"])

    V_bd = data_file["V_bd"].iloc[0]
    poly_coefs = data_file["coefs"].iloc[0]

    def norm_derivative(x, y):
        dy_dx = np.gradient(y) / np.gradient(x)
        return 1 / y * dy_dx

    derivative = norm_derivative(x, y)
    y_poly = (
        poly_coefs[0]
        + poly_coefs[1] * x
        + poly_coefs[2] * x**2
        + poly_coefs[3] * x**3
        + poly_coefs[4] * x**4
        + poly_coefs[5] * x**5
    )
    x_gauss = x[
        np.logical_and(
            x >= (V_bd - data_file["width"].iloc[0] / 2),
            x <= (V_bd + data_file["width"].iloc[0] / 2),
        )
    ]

    def gauss(x, H, A, mu, sigma):
        return H + A * np.exp(-((x - mu) ** 2) / (2 * sigma**2))

    y_gauss = gauss(x_gauss, *data_file["params"].iloc[0])

    sipm_number = list(data_file["SiPM"].drop_duplicates())[0]

    fig, (ax, ax2) = plt.subplots(
        2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    fig.suptitle(f"Reverse IV curve: SiPM {sipm_number}")
    ax.set_ylabel("Current(mA)")
    ax2.set_ylabel(r"$I^{-1} \frac{dI}{dV}$", color="darkgreen")
    ax2.tick_params(axis="y", colors="darkgreen")

    ax.errorbar(
        data_file["V"], data_file["I"], data_file["I_err"], marker=".", label="Data"
    )
    ax.legend(loc="upper right")
    ax.grid(True)

    ax2.scatter(x, derivative, marker="o", s=5, color="darkgreen", label="Derivative")
    ax2.plot(x, y_poly, color="darkturquoise", label="5th-deg polynomial")
    ax2.plot(x_gauss, y_gauss, color="darkorange", label="Gaussian around peak")
    ax2.axvline(
        V_bd,
        color="gold",
        label=f"V_bd = {V_bd:.2f} ± {abs(data_file['V_bd_std'].iloc[0]):.2f} V",
    )
    ax2.legend(loc="upper left")

    pdf.savefig()
    plt.close()
