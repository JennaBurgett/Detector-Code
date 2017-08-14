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

file_name = raw_input("Please input Source data: ")       #prompts user to input a file name, must be in .csv format

	   #initializing the lists for each channel to be filled from file
V1 = []    #V1 = ADC value of peak
W1 = []	   #W1 = std deviation of ADC value of peak
U1 = []	   #U1 = Energy in keV of peak

with open(file_name) as csvfile:                       #appends each column of data into a 1D numpy array.
    readCSV = csv.reader(csvfile, delimiter=',')

    for row in readCSV:
        V1.append(float(row[0]))
        W1.append(float(row[1]))
        U1.append(float(row[2]))
y = np.array(V1)
Sigma_y = np.array(W1)
x = np.array(U1)

#Defining the fitting function:

def linear_fit(x,a,b):
    return a + b*x

# Parameter Best Guess values
a0 = 1
b0 = 1

# Curve Fitting Function
popt, pcov = curve_fit(linear_fit, x, y, p0=[a0,b0], sigma=Sigma_y)

#unpacking the curve_fit outputs
da, db = [np.sqrt(pcov[j, j]) for j in range(popt.size)]
a, b = popt

# Calculating goodness of fit reduced Chi^2
resids = y - linear_fit(x, *popt)
redchisqr = ((resids / Sigma_y) ** 2).sum() / float(x.size - 2)


Test_Box_Settings = raw_input("enter the detector settings for this calibration. (Bias V,Amplification,Crystal Geometry,photodevice and size, etc.):")

#Plot 1 data and best fit
figure = plt.figure("Calibration", figsize=(10,10))
gs = gridspec.GridSpec(2, 1, height_ratios=[6, 2])
axes1 = figure.add_subplot(gs[0])
axes1.errorbar(x, y, yerr=Sigma_y, ecolor="black", fmt="ro")
axes1.plot(x,linear_fit(x, *popt), lw=2, color="blue", label="Raw")
#axes1.plot(x, linear_fit(x, a+da, b), lw=2, color="green", label="High Offset Error")
#axes1.plot(x, linear_fit(x, a-da, b), lw=2, color="orange", label="Low Offset Error")
#axes1.plot(x, linear_fit(x, a, b+db), lw=2, color="purple", label="High Slope Error")
#axes1.plot(x, linear_fit(x, a, b+db), lw=2, color="yellow", label="Low Slope Error")
axes1.set_title(Test_Box_Settings)
axes1.set_ylabel("ADC Channel", fontsize=15)
axes1.set_ylim(0, max(y)+200)
axes1.set_xlim(0,max(x)+200)
axes1.legend()
axes1.text(0.1, 0.85, '$f(x) = A + Bx$', transform=axes1.transAxes, fontsize=12)
axes1.text(0.1, 0.75, 'A = {0:0.1f}$\pm${1:0.1f}'.format(a, da), transform=axes1.transAxes, fontsize=10)
axes1.text(0.1, 0.65, 'B = {0:0.5f}$\pm${1:0.5f}'.format(b, db), transform=axes1.transAxes, fontsize=10)
axes1.text(0.1, 0.55, '$\chi_r^2$ = {0:0.2f}'.format(redchisqr), transform=axes1.transAxes, fontsize=10)


# Plot 2 Goodness of Fit
axes2 = figure.add_subplot(gs[1])
axes2.errorbar(x, resids, yerr=Sigma_y, ecolor="black", fmt="ro")
axes2.axhline(color="gray", zorder=-1)
axes2.set_xlabel('Energy KeV', fontsize=15)
axes2.set_ylabel('residuals', fontsize=15)
axes2.set_xlim(0,max(x)+200)
axes2.set_ylim(-100, 100)
axes2.set_yticks((-100, 0, 100))

print("Best Fit Parameters:[A,B]")
print(a, b)
print("errors in fit parameters:[da,db]")
print(da, db)
print("Reduced Chi^2")
print(round(redchisqr, 3))


plt.show()

saveplot = raw_input("Do you wish to save the previous plot? y/n:")
if saveplot == "y":
    image_name1 = raw_input("save plot as? (filename.png):")
    figure.savefig(image_name1)