import claro_class as cl
import pandas as pd
import os
from tqdm import tqdm

# Get input path from user
input_path = input("Enter a file path or a single file: ")

# Check if input path exists
if not os.path.exists(input_path):
    print(f"The file/folder does not exist.")
    exit()

if os.path.isfile(input_path):
    # Check if input file is a txt file
    if not input_path.endswith(".txt"):
        print(f"This is not a .txt file.")
        exit()

    print(f"This is a single file. Analyzing...\n")
    try:
        single_file = cl.LinearFit(input_path)
        single_file.print_linear_data()
    except Exception as e:
        print(f"Error analyzing:")
        print(e)

    try:
        single_file = cl.ErrorFunctionFit(input_path)
        single_file.print_erf_data()
    except Exception as e:
        print(f"Error analyzing:")
        print(e)
    try:
        single_file = cl.HarryPlotter(input_path)
        single_file.plotter()
    except Exception as e:
        print(f"Error analyzing:")
        print(e)
else:
    print(f"This is a folder. Analyzing...\n")
    try:
        folder = cl.FolderReader(input_path)
        folder.read_folder()
    except Exception as e:
        print(f"Error analyzing:")
        print(e)

    try:
        with open(f"matching_files.txt", "r") as file:
            paths = file.readlines()
            paths = [path.strip() for path in paths]
            data_processed = []

            for path in tqdm(paths, "Analyzing files"):
                single_file = cl.DataReader(path, data_processed)
                single_file.data_reader()

            processed_dataframe = pd.DataFrame(
                data_processed,
                columns=[
                    "station",
                    "chip",
                    "channel",
                    "amplitude",
                    "width",
                    "transition_point_data",
                    "transition_point_erf",
                    "std_transition_point_(erf)",
                ],
            )

            processed_dataframe.to_csv(f"{os.getcwd()}/results.csv", index=False)
            single_file = cl.Histogram(processed_dataframe)
            single_file.create_histogram()

    except Exception as e:
        print(f"Error analyzing:")
        print(e)
        exit()
