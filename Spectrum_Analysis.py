
import matplotlib.pyplot as plt
import numpy as np
import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array
from matplotlib.pyplot import plot, scatter, show
from scipy.optimize import curve_fit
import math
import matplotlib.gridspec as gridspec

#IMPORTING DATA AND CREATING THE PROPER ARRAY'S
import csv

file_name = 'AM241_9x9x40_SiPM_28V_4usSHAPE_60%live_cap.csv'        #Copy and paste the data into this spot in quotations

adc = []                                           #initializing the lists for each channel to be filled from file
counts = []
with open(file_name) as csvfile:                       #appends each column of data into a 1D numpy array.
    readCSV = csv.reader(csvfile, delimiter=',')

    for row in readCSV:
        adc.append(float(row[0]))
        counts.append(float(row[1]))
Channel = np.array(adc)
Counts1 = np.array(counts)
Energy = Channel#/14.829028453115578 - -20.779719147123696/14.829028453115578   #Using the equation from the calibration code to change Channel into Energy

#Making a while loop that continues the plotting until the user says no to manually entering parameters
cont = True
while cont == True:
    
#Prompts the user what the desired parameters are for the plot 
    CountRange = input("What is the maximum counts?")
    EnergySmall = input("What is the minimum energy?")
    EnergyBig = input("What is the maximum energy?")
    Frequency = input("What is the tick frequency?")
    
#Changing the answers into float values since that is required for set_xticks
    EnergyMax = float(EnergyBig)
    EnergyMin = float(EnergySmall)
    Tick_Frequency = float(Frequency)
    ticks = np.arange(EnergyMin, EnergyMax, Tick_Frequency)

#Plotting the data using the parameters given by the user above
    figure1 = plt.figure(file_name, figsize = (7,5))
    axes1 = plt.subplot(111)
    axes1.plot(Energy, Counts1, color="black", lw=0.5)
    axes1.set_title("AM241, 9x9x40, 28V", fontsize=20)
    axes1.set_xlabel("ADC Bins", fontsize=20)
    axes1.set_ylabel("Counts", fontsize=20)
    axes1.set_ylim([1, CountRange])
    axes1.set_xlim([EnergyMin, EnergyMax])
    axes1.set_xticks(ticks)
    axes1.grid()
    axes1.fill_between(Energy, Counts1)
    plt.show()

    saveplot = raw_input("Do you wish to save the previous plot? y/n:")
    if saveplot == "y":
        image_name1 = raw_input("save plot as? (filename.png):")
        figure1.savefig(image_name1)
    
    redo_params = raw_input("Do you want to manually enter the parameters?")
    if redo_params == "n":
        cont = False