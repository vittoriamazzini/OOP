import sipm_class as cl
import os


input_path = input("Enter a file path or a single file: ")
if not os.path.exists(input_path):
    print(f"The file/folder does not exist.")
    exit()

if os.path.isfile(input_path):
    if not input_path.endswith(".csv"):
        print(f"This is not a .csv file.")
        exit()

    print(f"This is a single file. \n")
    print(f"Analyzing one single file... \n")
    single_file = cl.SingleFile(input_path)
    single_file.file_reader()
    single_file.single_analyzer()
else:
    print(f"This is a folder. \n")
    exit()
