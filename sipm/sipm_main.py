import sipm_class as cl
import os

# Get input path from user
input_path = input("Enter a file path or a single file: ")

# Check if input path exists
if not os.path.exists(input_path):
    print("The file/folder does not exist.")
    exit()

if os.path.isfile(input_path):
    # Check if input file is a CSV file
    if not input_path.endswith(".csv"):
        print("This is not a .csv file.")
        exit()

    print(f"This is a single file. Analyzing...\n")

    # Analyze single file using sipm_class.SingleFile
    try:
        sipm = cl.SingleFile(input_path)
        sipm.file_reader()
        sipm.single_analyzer()
    except Exception as e:
        print(f"Error analyzing:")
        print(e)

else:
    print(f"This is a folder. Analyzing...\n")

    # Analyze folder using sipm_class.MultipleFiles
    try:
        sipm = cl.MultipleFiles(input_path)
        sipm.read_folder()
        sipm.dir_analyzer()
        sipm.create_histogram()
    except Exception as e:
        print("Error analyzing:")
        print(e)


# C:\Users\utente\Desktop\OOP\CACTUS_HPK_measurements
# C:\Users\utente\Desktop\random
