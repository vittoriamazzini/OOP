# OOP

## Final project for the Object-Oriented Programming course

This repository contains the final project for the OOP course at the University of Ferrara. The main goal of the course was to learn the basics of the OOP paradigm and how to apply it in C++ and Python.

The project consists of two different assignments:

1) Claro: Read all the files in the folder "secondolotto_1", find the good ones, extract the data about the transition point, width and store the ADC and Counts into two arrays. Via the Least Square Method on the transition zone (counts != 0 && counts !=1000) evaluate the transition point and compare it with the one present in the file. Use the erf function (SciPy module) to fit the points and evaluate again the transition point and width. Graph an histogram of the discrepancies between the expected and evaluated values.
2) SiPM: Evaluate the direct and reverse curves of the SiPM data in the "CACTUS_HPK" folder. Analyze the peaks of each waveform. Estimate the dark count rate.

## How to use
Download the *_main.py and *_class.py and store them in a folder. The *_main.py file is the "front end" of the program, change the various function arguments here and then run the program via terminal, specifying the file/directory path as the first argument.
