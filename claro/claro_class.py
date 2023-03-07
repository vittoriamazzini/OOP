import fnmatch
import re
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy import stats, optimize, special
import matplotlib.pyplot as plt
import warnings


class FolderReader:
    """Reads files in a folder and returns the matching files."""

    def __init__(self, root_folder):
        self.root_folder = root_folder

    def read_folder(self):
        """Walks through the folder and returns a list of matching folders and files."""
        matching_files = []
        bad_files = []

        for root, dirs, filenames in tqdm(os.walk(self.root_folder), "Reading files"):
            if fnmatch.fnmatch(
                root, "*Station_1__*/Station_1__??_Summary/Chip_*/S_curve"
            ):
                for filename in filenames:
                    if fnmatch.fnmatch(filename, "Ch_*_offset_*_Chip_*.txt"):
                        file_path = os.path.abspath(os.path.join(root, filename))
                        with open(file_path) as f:
                            first_line = f.readline()
                            if re.search("[a-zA-Z]", first_line):
                                bad_files.append(file_path)
                                continue
                            matching_files.append(file_path)

        with open("matching_files.txt", "w") as f:
            for file_path in matching_files:
                f.write(file_path + "\n")
        with open("bad_files.txt", "w") as f:
            for file_path in bad_files:
                f.write(file_path + "\n")
        with open("all_files.txt", "w") as f:
            for file_path in matching_files + bad_files:
                f.write(file_path + "\n")
        return {"matching_files": matching_files, "bad_files": bad_files}


class LinearFit:
    """Performs linear regression on the data in a file.

    The data is stored in the instance variables `height`, `tr_point`, `width`, `x`,
    `y`, `_metadata`, and `fit_guess`.

    Args:
        path (str): The path to the data file.

    """

    def __init__(self, path):
        """Initialize the class.

        Parameters:
        path (str): The file path for the data.

        """
        self.path = path
        data = _get_data(path)
        self.height = data["height"]
        self.tr_point = data["tr_point"]
        self.width = data["width"]
        self.x = data["x"]
        self.y = data["y"]
        self.meta = data["meta"]
        self.linear_params = {}

    def linear_fit(self):
        """Performs linear regression on the data and returns the results.

        Returns:
            dict: A dictionary with the following keys:
                - "slope"
                - "intercept"
                - "transition point (linear)"
                - "R_squared"
        """
        x_fit = self.x
        y_fit = self.y

        # removing duplicates values
        mask = np.ones(len(self.y), dtype=bool)
        mask[np.where(self.y[:-1] == self.y[1:])[0]] = False
        self.x_fit, self.y_fit = self.x[mask], self.y[mask]

        self.x_fit = x_fit
        self.y_fit = y_fit

        model = stats.linregress(x_fit, y_fit)
        half_max_height = (y_fit.max() - y_fit.min()) / 2
        self.fitted_trans_point = (half_max_height - model.intercept) / model.slope

        self.linear_params = {
            "slope": model.slope,
            "intercept": model.intercept,
            "half maximum height": half_max_height,
            "transition point (linear)": self.fitted_trans_point,
            "R_squared": model.rvalue**2,
        }
        return self.linear_params

    def print_linear_data(self):
        print(f"The data obtained from the file is:")
        for key, value in self.meta.items():
            print(f"{key}: {value}")
        print("\n")
        print(f"The data obtained from the linear fit is:")
        for key, value in self.linear_fit().items():
            print(f"{key}: {value}")
        print("\n")


class ErrorFunctionFit:
    """A class for fitting data to an error function.

    Attributes:
        height (float): The height of the error function.
        tr_point (float): The transition point of the error function.
        width (float): The width of the error function.
        x (numpy.ndarray): The x-coordinate data.
        y (numpy.ndarray): The y-coordinate data.
        meta (dict): Additional metadata about the data.
        fit_guess (list): The initial guess for the fit parameters.
        fit_params (numpy.ndarray): The fit parameters for the error function.
        erf_x_fit (numpy.ndarray): The x-coordinate data for the fit.
        erf_y_fit_fit (numpy.ndarray): The y-coordinate data for the fit.

    """

    def __init__(self, path):
        """Initialize the class.

        Parameters:
            path (str): The file path for the data.

        """
        data = _get_data(path)
        self.height = data["height"]
        self.tr_point = data["tr_point"]
        self.width = data["width"]
        self.x = data["x"]
        self.y = data["y"]
        self.meta = data["meta"]
        self.fit_guess = data["fit_guess"]
        self.erf_params = {}
        self.fit_params = None
        self.erf_x_fit = None
        self.erf_y_fit = None

    def erf_fit(self, fit_guess=None):
        """Fit the data to an error function and return the fit parameters.

        Parameters:
            fit_guess (list, optional): The initial guess for the fit parameters.
                If None, use the fit_guess from the class initialization.

        Returns:
            dict: The fit parameters for height, transition point (erf), and width.
                The values are in the form [fit_value, standard_deviation].

        """
        x = self.x
        y = self.y

        if fit_guess is None:
            fit_guess = self.fit_guess

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message="Covariance of the parameters could not be estimated"
            )
            warnings.filterwarnings(
                "ignore", message="invalid value encountered in sqrt"
            )

            fit_params, covar = optimize.curve_fit(
                modified_erf, x, y, fit_guess, maxfev=10000
            )

            std = np.sqrt(np.diag(covar))

        erf_x_fit = np.linspace(self.x.min(), self.x.max(), 100)
        erf_y_fit = modified_erf(erf_x_fit, *fit_params)

        if np.isinf(std[1]):
            std[0] = np.nan
            std[1] = np.nan
            std[2] = np.nan

        self.erf_params = {
            "height": [fit_params[0], std[0]],
            "transition point (erf)": [fit_params[1], std[1]],
            "width": [fit_params[2], std[2]],
            "erf_x": erf_x_fit,
            "erf_y": erf_y_fit,
        }
        return self.erf_params

    def print_erf_data(self):
        print(f"The data obtained from the erf fit is:")
        for key, value in self.erf_fit().items():
            if key != "erf_x" and key != "erf_y":
                print(f"{key} : {value}")
        print("\n")


class DataReader:
    """Reads data from a file and processes it to extract relevant information."""

    def __init__(self, path, data_processed):
        """
        Initializes the DataReader instance.

        Params:
        path: Path to the file from which the data is to be read.
        data_processed: A list where the processed data will be appended.
        """
        self.path = path
        self.data = _get_data(path)
        self.fileinfo = _get_fileinfo(path)
        self.erf = ErrorFunctionFit(path).erf_fit()
        self.data_processed = data_processed

    def data_reader(self):
        """
        Reads data from a file, processes it, and appends the processed data to the data_processed list.

        Return: None
        """
        row = [
            self.fileinfo["station"],
            self.fileinfo["channel"],
            self.fileinfo["chip"],
            self.data["width"],
            self.data["height"],
            self.data["tr_point"],
            self.erf["transition point (erf)"][0],
            self.erf["transition point (erf)"][1],
        ]

        self.data_processed.append(row)


class Histogram:
    """Creates a histogram of the transition points distribution.

    Args:
        processed_dataframe (DataFrame): DataFrame containing the processed data.
    """

    def __init__(self, processed_dataframe):
        self.df = processed_dataframe

    def create_histogram(self, saveplot=True):
        """Creates a histogram of the transition points distribution.

        Args:
            saveplot (bool, optional): Boolean indicating whether the plot should be saved or not. Defaults to True.

        Returns:
            None
        """
        transition_points_erf = self.df.transition_point_erf
        transition_points_data = self.df.transition_point_data
        discrepancy = transition_points_data - transition_points_erf

        fig, axs = plt.subplots(3)
        fig.suptitle("Histogram of the transition points distribution")

        axs[0].hist(transition_points_data, bins=200, color="coral")
        axs[1].hist(transition_points_erf, bins=200, color="limegreen")
        axs[2].hist(discrepancy, bins=20, range=(-1e-6, 1e-6), color="cadetblue")

        axs[0].set_title("Transition points (data)")
        axs[1].set_title("Transition points (erf)")
        axs[2].set_title("Discrepancy")

        plt.tight_layout()

        if saveplot == True:
            plotname = f"Histogram_transition_points.png"
            plt.savefig(plotname, bbox_inches="tight")
            print(f"Plot saved as {os.getcwd()}\{plotname}")
        plt.show()


class HarryPlotter:
    """
    Class for plotting data with linear and error function fits.

    Attributes:
        path (str): Path to the data file.
        fileinfo (dict): Dictionary containing information about the file.
        data (dict): Dictionary containing the data points to be plotted.
        linear (LinearFit): LinearFit object for linear fit of the data.
        erf (ErrorFunctionFit): ErrorFunctionFit object for error function fit of the data.
    """

    def __init__(self, path):
        """
        Initialize the class.

        Parameters:
        path (str): The file path for the data.

        """
        self.path = path
        self.fileinfo = _get_fileinfo(path)
        self.data = _get_data(path)
        self.linear = LinearFit(path)
        self.erf = ErrorFunctionFit(path)

    def plotter(self, saveplot=True):
        """
        Plot the data with linear and error function fits.

        Parameters:
            saveplot (bool, optional): Boolean value indicating whether to save the plot or not. Default is True.

        Returns:
            None
        """
        fig, ax = plt.subplots()

        fig.suptitle(
            f"Fit Claro: Station {self.fileinfo['station']}, Chip {self.fileinfo['chip']}, Channel {self.fileinfo['channel']}"
        )
        plt.plot(self.data["x"], self.data["y"], "o", label="data")
        plt.plot(
            self.linear.x,
            self.linear.linear_fit()["slope"] * self.linear.x
            + self.linear.linear_fit()["intercept"],
            "g-",
            label="linear fit",
        )
        plt.plot(
            self.erf.erf_fit()["erf_x"],
            self.erf.erf_fit()["erf_y"],
            "-",
            label="erf fit",
        )
        plt.scatter(
            self.linear.linear_fit()["transition point (linear)"],
            self.linear.linear_fit()["half maximum height"],
            color="k",
            marker="+",
            label="linear transition point",
        )
        plt.scatter(
            self.erf.erf_fit()["transition point (erf)"][0],
            self.linear.linear_fit()["half maximum height"],
            color="k",
            marker="^",
            label="erf transition point",
        )
        plt.xlabel("ADC")
        plt.ylabel("Counts")
        plt.legend()

        if saveplot == True:
            plotname = f"Plot_Claro_Chip{self.fileinfo['chip']}_Ch{self.fileinfo['channel']}.png"
            plt.savefig(plotname, bbox_inches="tight")
            print(f"Plot saved as {os.getcwd()}\{plotname}")
        plt.show()


######################## Static Methods ##############################


@staticmethod
def modified_erf(x, height, a, b):
    """
    Computes the modified error function.

    Parameters:
    x (array): The x-axis values.
    height (float): Amplitude of the function.
    a (float): The center of the function.
    b (float): The width of the function.

    Returns:
    array: The y-axis values of the modified error function.
    """
    return height / 2 * (1 + special.erf((x - a) / (b / 2 * np.sqrt(2))))


def _get_fileinfo(path):
    """
    Retrieves Station chip and channel number from the path.

    Parameters:
    path (str): The file path.

    Returns:
    dict: A dictionary containing the station name, chip number and channel number.
    """
    try:
        _chip = re.search(".+Chip_(.+?).txt", path).group(1)
        _channel = re.search(".+Ch_(.+?)_.+", path).group(1)
        _station = re.search(".+\Station_1__(.+?)_Summary.+", path).group(1)
    except AttributeError:
        _station = "?"

    _fileinfo = {"station": _station, "chip": _chip, "channel": _channel}
    return _fileinfo


def _get_data(path):
    """
    Retrieves data from a file and stores it in a dictionary.

    Parameters:
    path (str): The file path.

    Returns:
    dict: A dictionary containing the data, meta information, and fit guess.
    """
    data = pd.read_csv(path, sep="\t", header=None, skiprows=None)
    height = data[0][0]
    tr_point = data[1][0]
    width = np.abs(data[2][0])
    x = data.iloc[2:, 0].to_numpy()
    y = data.iloc[2:, 1].to_numpy()

    _metadata = {
        "path": path,
        "amplitude": height,
        "transition point": tr_point,
        "width": width,
    }
    fit_guess = [height, tr_point, width]

    all_data = {
        "height": height,
        "tr_point": tr_point,
        "width": width,
        "x": x,
        "y": y,
        "meta": _metadata,
        "fit_guess": fit_guess,
    }
    return all_data
