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
    sipm = cl.SingleFile(input_path)
    sipm.file_reader()
    sipm.single_analyzer()
else:
    print(f"This is a folder. \n")
    sipm = cl.MultipleFiles(input_path)
    sipm.read_folder()
    sipm.dir_analyzer()
    sipm.create_histogram()

# C:\Users\utente\Desktop\OOP\CACTUS_HPK_measurements
# C:\Users\utente\Desktop\random
