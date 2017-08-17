"""
===========================================================
= READ ME NOW GO NO FURTHER UNTIL THIS MISSION IS COMPLETE=
===========================================================

Make sure that the lines starting from import csv to Channel = np.array(V1) are uncommented
and comment away the lines starting from sigma = __ to Channel = np.append (Work, Ha). This
will make sure that it can read the .csv file that is coming from the ADC board. (Make
sure the file is saved as a CSV(delimiter) type, with exactly that wording or else it 
won't work).

The file you wish to use must be in the same location as this code. In order to use the desired
data, you must type the name of the file exactly (the entire name) when prompted with "Please 
input Source data:", so it is usually easier to just copy and paste the name.

General Rules:
    - Always reply with y or n, do not type anything else unless it asks you a value or a name.
        - In most cases if you type anything other than y, it will just continue, but NOT all
        cases, so try to answer every question.
    - This is ultimately a guessing game at the end, so sometimes the parameters you give will
    lead to an error and cause you to restart, THAT IS OK DON'T YOU PANIC, the ok errors will
    be listed down below in the "what all this does" section.
    - If you don't know what a variable is or are looking for something specific, it IS POSSIBLE
    to use ctrl + f to search (at least using Spyder because it is wonderful).
    - If you need more in depth help with understanding what is going on in the curve fitting section
    of the script, most of the helpful comments are in the first portion of it, in the first Gaussian
    option, where it is just Gaussian with nothing added.
        - Also, the curve_fit function is something that was made through scipy. They have source code
        there if you want a more detailed explanation for what it does.
    

What all this does:
    1. Plots all the raw data in ADC bins vs. index (index is put in place to mimick the time
    until the real time per data point is given and then it will be replaced) along with the peakfinding
    funciton.
    
    2. The peakfinding function will allow us to differentiate between points we would want
    and points that may be in the noise floor. The peakfinding function will spit out a matrix
    of [index, ADC bin].
    
    3. The output of the peakfinding function is then made into a list of all ADC bins and 
    put into a histogram function that organizes the ADC bins into a plot of Counts vs ADC bins
    and outputs bins and the corresponding counts.
    
    4. The bins and counts that comes from the histogram plot is then used to plot bins and counts as
    well as the peakfinding function to find the peak of the final spectrum.
    
    5. You will then be asked if you want to make a curve fit, the greatest honor you will ever recieve,
    so usually say ye if you so desire. There are four different options for fits, they are all different
    forms of gaussian starting at plain gaussian and and adding on until it reaches third order plus gaussian.
    It will ask for the expected energy, which is the energy that others have already discovered and can
    be found on the google. BUT when it asks for the ADC bin, that is printed out by the peakfinding function
    above the plot made in step 4 of data from the histogram (now is green) and the black dots from the
    peakfinding function, it is important to pick the point that corresponds with the peak you are trying
    to fit and that it is the exact number printed. It will also ask for a width, just make a good guess.
    If you are given an error at this point, if it says "Optimal parameters not found: Number of calls to 
    function has reached maxfev = _____", then that means that the width was incorrect, it is either too 
    large or too small, the number given for maxfev doesn't seem to correlate with whether it is too large
    or too small at all. The curve fitting is all a guessing game, just keep guessing until you get something 
    that works best for what you are doing, wether it be which one gives the best resolution with certain parameters
    or what fits the best regardless of the resolution, make sure to try as many different combinations as you can,
    sometimes what works best can surprise you. Once you get what you want, you then adress the couple questions after
    the curvefit plot is made, if you want to save it, and if you want to save the pertinent data for calibration.
    If you have found the perfect plot for you then say "y" to the first question, and that plot you just made will
    be saved in the same location as this code. It asks you for a name afterwards and continues with the code.
    Usually I just save the image with the same name as the data file so you can keep track of what data file
    this image came from. For the second save question "do you wish to save the pertinent data for calibration",
    only say y if you are hoping to calibrate.
    
    6. If say y to "do you wish to save the pertinent data for calibration", a .csv file will be created
    called calibration_data.csv. What the code does is save three data points from the plot you made, the 
    the expected energy you gave while making the plot, the ADC bin that it found the peak was at, and the
    deviation in the ADC bin it found the peak was at. These data points are put in a row in the .csv file
    created. If the file calibration_data.csv exists already, it will keep adding the data points to the 
    next row when you say y to the question again. If you wish to calibrate, keep saying y to the question
    for each plot you wish to use to calibrate and make sure that the file calibration_data.csv is in the 
    same location as this code.
        *** In order to actually do the calibration, there is a different script to use called 
            Calibration27.py, there you will use the file created (calibration_data.csv) to
            calibrate.
            
YAY you completed reading this, you can now continue and thank the lords that there is no more 
long reading

....unless something goes wrong :)
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array
from matplotlib.pyplot import plot, scatter, show
from scipy.optimize import curve_fit
import math
import matplotlib.gridspec as gridspec

def smoothListGaussian(list,degree=5):
#smooth_list_Gaussian smooths a data set by averaging using a weigthed Gaussian
#It takes the previous 4 points and averages slopes.
#Returns a smoothed 1D array of identical size of input data used to better identify peaks.

     window=degree*2-1

     weight=np.array([1.0]*window)

     weightGauss=[]

     for i in range(window):

         i=i-degree+1

         frac=i/float(window)

         gauss=1/(np.exp((4*(frac))**2))

         weightGauss.append(gauss)

     weight=np.array(weightGauss)*weight

     smoothed=[0.]*(len(list)-window)

     for i in range(len(smoothed)):

         smoothed[i]=sum(np.array(list[i:i+window])*weight)/sum(weight)

     return smoothed

def peakdet(v, delta, x=None):
    """
    Returns two arrays

    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was followed by a value lower by
    %        DELTA.        sys.exit('Input vectors v and x must have same length')

    if not isscalar(delta):

    """
    maxtab = []
    mintab = []

    if x is None:
        x = arange(len(v))

    v = asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN

    lookformax = True

    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return array(maxtab), array(mintab)

#gauss_functions are used to fit a curve to the data set using pysci curve_fit
def gauss_function(x, P, xmean, s):
    return P * np.exp(-(x - xmean) ** 2 / (2 * s ** 2))

def firstorderpoly_gauss_function(x, a, b, P, xmean, s):
    return a + b * x + P * np.exp(-(x - xmean) ** 2 / (2 * s ** 2))

def secondorderpoly_gauss_function(x, a, b, c, P, xmean, s):
    return a + b * x + c * x * x + P * np.exp(-(x - xmean) ** 2 / (2 * s ** 2))

def thirdorderpoly_gauss_function(x, a, b, c, d, P, xmean, s):
    return a + b * x + c * x * x + d * x * x * x + P * np.exp(-(x - xmean) ** 2 / (2 * s ** 2))


#IMPORTING DATA AND CREATING THE PROPER ARRAY'S
import csv

"""file_name = raw_input("Please input Source data: ")       #prompts user to input a file name, must be in .csv format
V1 = []              
                            #initializing the lists for each channel to be filled from file
with open(file_name) as csvfile:                       #appends each column of data into a 1D numpy array.
    readCSV = csv.reader(csvfile, delimiter=',')

    for row in readCSV:
        V1.append(float(row[0]))
Channel = np.array(V1)
"""

#Since the format of the ADC data is that every third row is a data packet, here we pick out those rows and turn them into an array for us to use.
file_name=raw_input("input file name: ")

val=[]
i=0
with open(file_name) as csvfile:
    readCSV=csv.reader(csvfile, delimiter= ',')
    for row in readCSV:
        print "beginning"
        if i==0:
            print "yo"
            i=i+1
        if i % 3 == 0:
            print "we made it doggie"
           # print row
            for j in arange(100):
              #  print row[j]
                val.append(float(row[j]))
            i=i+1
        else:
            i=i+1
        print "ending one row in csv"
        
Channel = np.array(val)

#Creates random numbers in order to test the code, the sigma changes how wide it gets while the
#added on portion shifts it up in ADC bins.
#sigma = 51
#Work = sigma * np.random.randn(1000000) + 602
#sigma=50
#Ha = sigma*np.random.randn(1000000) + 165
#Channel = np.append (Work, Ha)

#naming radiation source, this is used later as the title of the plot
Source = raw_input("Enter name of Radiation Source:")


# Finding Peaks of the 'smoothed' data set using the peakdet function defined above
#The Delta is set at 1 just to start with, and is possible to change later when asked to manually change parameters.
Delta = 1
series = Channel
maxtab, mintab = peakdet(series, Delta)

#Adding to the channel range (y-axis) just makes it so you can see the plot a little better, might change to less than 100 if it seems to be too large.
ChannelRange = max(Channel)+100

#Initial plot of data that plots the raw data from the ADC as ADC bin vs Index (eventually change index to time) as well as the results from the peakdet function.
figure1 = plt.figure(file_name)
axes1 = plt.subplot(111)
axes1.plot(Channel, color="green", lw=0.5, label="Raw")
axes1.scatter(array(maxtab)[:, 0], array(maxtab)[:, 1], color='black')
axes1.set_title(Source, fontsize=20)
axes1.set_xlabel("Index", fontsize=20)
axes1.set_ylabel("Channel", fontsize=20)
axes1.set_ylim([1, ChannelRange])
axes1.grid()
axes1.legend()

#Prints out what maximums the peakdet function found
print("Peaks at:")
print("[Index, ADC Bin]")
print(maxtab)
plt.show()

#If the plot is too wide or you want to change the results of the peakdet function, you can do that here by replying "y" and answering the prompts.
Manual_Params = raw_input("Do you wish to manually re-enter the plotting parameters? y/n:")
if Manual_Params=="y":
    while True:
        ChannelRange = input("Enter Maximum Channel value:")
        IndexMin = input("Enter minimum x value:")
        IndexMax = input("Enter maximum x value:")
        Delta = input("Enter minimum peak: ")
        
        series = Channel
        maxtab, mintab = peakdet(series, Delta)

        figure1 = plt.figure(file_name)
        axes1 = plt.subplot(111)
        axes1.plot(Channel, color="green", lw=0.5, label="Raw")
        axes1.scatter(array(maxtab)[:, 0], array(maxtab)[:, 1], color='black')
        axes1.set_title(Source, fontsize=20)
        axes1.set_xlabel("Index", fontsize=20)
        axes1.set_ylabel("ADC Channel", fontsize=20)
        axes1.set_ylim([1, ChannelRange])
        axes1.set_xlim([IndexMin, IndexMax])
        axes1.grid()
        axes1.legend()
        
        print("Peaks at:")
        print("Index, ADC Bin")
        print(maxtab)
        plt.show()
        redo_params = raw_input("Do you wish to re-enter the plotting parameters? y/n:")
        if redo_params == "n":
            break

#Allows you to save the plot that was just made
saveplot = raw_input("Do you wish to save the previous plot? y/n:")
if saveplot == "y":
    image_name1 = raw_input("save plot as? (filename.png):")
    figure1.savefig(image_name1)
    
maxtab1 = maxtab[:,1]   #Specifies the ADC side of what the peakfinding function output as a single array

#makes the histogram, this function also outputs the counts and in what ADC bins they were in
#This is set up for it to plot the ADC bins for the bins of the histogram and the bin range is from the minimum to maximum of the ADC bins (+1 max to see the data clearer) with a bin width of 1.
counts, bins, bars = plt.hist(maxtab1, 1000)
      
plt.xlabel("ADC Bins")
plt.ylabel("Counts")
plt.title(Source)
plt.show()

#will allow you to also change not only the range of the axis but also change the bin width.
Manual_Params = raw_input("Do you wish to manually re-enter the plotting parameters? y/n:")
if Manual_Params=="y":
    while True:
        bin_number = input("Enter number of bins: ")
        CountRange = input("Enter Maximum Counts value:")
        ChannelMin = input("Enter minimum ADC Channel:")
        ChannelMax = input("Enter maximum ADC Channel:")
        
        counts, bins, bars = plt.hist(maxtab1, bin_number)
        
        axes1 = plt.subplot(111)
        axes1.set_ylim([1, CountRange])
        axes1.set_xlim([ChannelMin, ChannelMax])
        axes1.set_xlabel("ADC Bins")
        axes1.set_ylabel("Counts")
        axes1.set_title(Source)
        plt.show()
        
        redo_params = raw_input("Do you wish to re-enter the plotting parameters? y/n:")
        if redo_params == "n":
            break
        
#the counts given by the histogram function is always one less than the bins given because the bins include an extra data point with no counts, so adding a zero is all that's needed to be able to plot it.
counts1 = np.append(counts, [0])

Delta = input("enter minimum height of expected peaks (required):")
series = counts1
maxtab, mintab = peakdet(series, Delta)

CountRange = max(counts1)+100
ChannelMin = min(bins)
ChannelMax = max(bins)

#Will plot ADC bins vs counts that was given in the histogram plot as well as the maximum values found by the peakdet function as scattered points
figure1 = plt.figure(file_name)
axes1 = plt.subplot(111)
axes1.plot(bins, counts1, color="green", lw=0.5, label="Raw")
#kickass is an array of integers of the index values found by the peak function
kickass = maxtab[:,0].astype(int)
#Here kickass is used as an indexing method for the bins that are given by the histogram function.
#Since the peakdet function uses the counts1, it uses counts1 vs indexes, so by using the output "ADC values" (really index values), it is possible to find the matching bin with the counts by using indexing on bins
axes1.scatter(bins[kickass], array(maxtab)[:, 1], color='black')
axes1.set_title(Source, fontsize=20)
axes1.set_xlabel("ADC Values", fontsize=20)
axes1.set_ylabel("Counts", fontsize=20)
axes1.set_ylim([1, CountRange])
axes1.set_xlim([ChannelMin, ChannelMax])
axes1.grid()
axes1.legend()

print("Peaks at:")
print("[ADC Bin,Count]")
print(np.column_stack((np.rint(array(bins[kickass])), maxtab[:,1])))
plt.show()

Manual_Params = raw_input("Do you wish to manually re-enter the plotting parameters? y/n:")
if Manual_Params=="y":
    while True:
        CountRange = input("Enter Maximum Counts value:")
        ChannelMin = input("Enter minimum ADC Channel:")
        ChannelMax = input("Enter maximum ADC Channel:")
        Delta = input("Enter minimum height of expected peaks (required): ")
        
        series = counts1
        maxtab, mintab = peakdet(series, Delta)
        
        kickass = maxtab[:,0].astype(int)

        # Manually entered plot parameters to look at a specific peak
        figure1 = plt.figure(file_name)
        axes1 = plt.subplot(111)
        axes1.plot(bins, counts1, color="green", lw=0.5, label="Raw")
        #axes1.plot(bins, smoothed, color="red", lw=1, label="Smoothed")
        axes1.scatter(bins[kickass], array(maxtab)[:, 1], color='black')
        axes1.set_title(Source, fontsize=20)
        axes1.set_xlabel("ADC Values", fontsize=20)
        axes1.set_ylabel("Counts", fontsize=20)
        axes1.set_ylim([1, CountRange])
        axes1.set_xlim([ChannelMin, ChannelMax])
        axes1.grid()
        axes1.legend()

        print("Peaks at:")
        print("[ADC Bin,Count]")
        #in order to print the values of the ADC bin and the Counts, I needed to mix the rounded maximum values with the counts
        print(np.column_stack((np.rint(array(bins[kickass])), maxtab[:,1])))
        plt.show()
        redo_params = raw_input("Do you wish to re-enter the plotting parameters? y/n:")
        if redo_params == "n":
            break

saveplot = raw_input("Do you wish to save the previous plot? y/n:")
if saveplot == "y":
    image_name1 = raw_input("save plot as? (filename.png):")
    figure1.savefig(image_name1)
    
#Curve fitting section
Gaussian_Fit = raw_input("Do you wish to fit a curve to the data? y/n:")


#The SciPy function Curve_fit uses the Levenberg-Marquardt method of fitting which is a combination of the
#steepest descent (or gradient) method and parabolic extrapolation.
#Note: the Levenberg-Marquardt method can fail if the initial guesses of the fitting parameters are too far away
#from the desired solution. This problem becomes more serious the greater the number of fitting parameters.
#Thus it is important to provide reasonable initial guesses for the fitting parameters and keep the number of parameters
#to the minimum necessary for a good fit.

if Gaussian_Fit == "y":
    while True:
        Gaussian = raw_input("Do you wish to fit a Gaussian? y/n:")
        if Gaussian == "y":
            Energy = raw_input("enter known energy for peak of interest in units of keV:")
            x0 = input("enter ADC Bin for peak of interest (from peak find output above):")
            n = input("enter width for region of interest (best guess):")
            #in order to give an approximate value for the 
            itemindex = np.where(np.rint(array(bins[kickass]))==x0)
            P0 = float(maxtab[itemindex[0],1])
            
            #pls represents what is going to be an array of bins within the range desired by the user
            pls = []
            #hlp represents what is going to be an array of counts within the range desired by the user
            hlp = []
            
            #The plot and the curve_fit function require data to plot that fits within the range that the user
            #dictates. This for loop does that by going through each point in bins to see if it is within a the range of the width given.
            for z in arange(len(bins)):
                if (bins[z] >= x0 - n/2 and bins[z] <= x0 + n/2):
                    pls.append(bins[z])
                    hlp.append(counts[z])
                    z = z + 1
                z = z + 1
            #makes data that is going to be plotted (x and y) all the data points in the range given by the for loop
            x = np.array(pls)
            y = np.array(hlp)
            #Error in counts is equal to the square root of the amount of counts.
            sigma_Counts = np.sqrt(hlp)
            #An equation of error for the curve_fit function to use as deviation.
            s0 = np.sqrt(sum((x-x0)**2)/(n-1))

            #Curve Fitting Function
            popt, pcov = curve_fit(gauss_function, x, y, p0=[P0, x0, s0],sigma=sigma_Counts)


            #using pcov in this way (one of the outputs for the curve_fit function) is able to reveal the deviation/errors by taking the square root of the diagonal
            dP, dxmean, ds = [np.sqrt(pcov[j, j]) for j in range(popt.size)]
            P, xmean, s = popt
            #Calculating the full width half max (FWHM) of the curve fit
            FWHM = 2 * np.sqrt(2 * np.log(2)) * abs(popt[2])
            #Calculating the resolution percentage of the curve fit
            Resolution = abs(FWHM / popt[1]) * 100

            # Calculating goodness of fit reduced Chi^2
            resids = y - gauss_function(x, *popt)
            redchisqr = ((resids / sigma_Counts) ** 2).sum() / float(x.size - 3)

            residplot = max(abs(resids))+float(0.3*max(abs(resids)))

            #There will be three different plots all in one image, a wide view of raw data and the curve fit, a zoomed in raw data and curve fit, and a residuals plot

            #Plot 1 wide view
            figure2 = plt.figure(file_name,figsize=(10,10))
            gs = gridspec.GridSpec(3, 1, height_ratios=[5, 6, 3])
            axes1 = figure2.add_subplot(gs[0])
            axes1.plot(bins, counts1, color="green", lw=0.5, label="Source Data")
            #axes1.plot(bins, smoothed, color="red", lw=0.5, label="Smoothed Data")
            #axes1.scatter(array(maxtab)[:, 0], array(maxtab)[:, 1], color='black')
            axes1.plot(x, gauss_function(x, *popt), lw=2, color="blue", label="Best Fit Curve")
            axes1.set_title((Source, Energy,"keV"), fontsize=20)
            axes1.set_ylabel("Counts", fontsize=15)
            axes1.set_ylim(1, max(gauss_function(x,*popt))+0.3*float(max(gauss_function(x,*popt))))
            axes1.set_xlim(0, 2047)
            #axes1.grid()
            axes1.legend(fontsize=10)

            #Plot 2 Zoomed into Peak
            axes2 = figure2.add_subplot(gs[1])
            axes2.plot(bins, counts1, color="green", lw=0.5)
            # axes2.plot(smoothed, color="red", lw=0.5)
            # axes2.scatter(array(maxtab)[:, 0], array(maxtab)[:, 1], color='black')
            axes2.plot(x, gauss_function(x, *popt), lw=2, color="blue")
            axes2.set_ylabel("Counts", fontsize=15)
            axes2.set_ylim(1, max(gauss_function(x,*popt))+0.3*float(max(gauss_function(x,*popt))))
            axes2.set_xlim([popt[1] - (n / 2 + 0.2 * n / 2), popt[1] + (n / 2 + 0.2 * n / 2)])
            #axes2.grid()
            axes2.text(0.1,0.85,'$g(x) = P\,e^{-(x-xmean)^2/2s^2}$', transform=axes2.transAxes,fontsize=12)
            axes2.text(0.1, 0.75,'P = {0:0.1f}$\pm${1:0.1f}'.format(P, dP),transform=axes2.transAxes,fontsize=10)
            axes2.text(0.1, 0.65,'xmean = {0:0.1f}$\pm${1:0.1f}'.format(xmean, dxmean),transform=axes2.transAxes,fontsize=10)
            axes2.text(0.1, 0.55,'s = {0:0.1f}$\pm${1:0.1f}'.format(s, ds),transform=axes2.transAxes,fontsize=10)
            axes2.text(0.75, 0.85,'$\chi_r^2$ = {0:0.2f}'.format(redchisqr),transform=axes2.transAxes,fontsize=10)
            axes2.text(0.75, 0.75, 'FWHM = {0:0.2f}'.format(FWHM), transform=axes2.transAxes, fontsize=10)
            axes2.text(0.75, 0.65, 'Resolution % = {0:0.2f}'.format(Resolution), transform=axes2.transAxes, fontsize=10)

            #Plot 3 Goodness of Fit
            axes3 = figure2.add_subplot(gs[2])
            axes3.errorbar(x, resids, yerr=sigma_Counts, ecolor="black", fmt="ro")
            axes3.axhline(color="gray", zorder=-1)
            axes3.set_xlabel('ADC Channel',fontsize=15)
            axes3.set_ylabel('residuals',fontsize=15)
            axes3.set_xlim([popt[1] - (n/2+0.2*n/2), popt[1] + (n/2+0.2*n/2)])
            axes3.set_ylim(-residplot,residplot)




            print("Best Fit Parameters:[P,mean,std_dev]")
            print(round(P,3),round(xmean,3),round(s,3))
            print("errors in fit parameters:[dP,dmean,dstd_dev]")
            print(round(dP,3),round(dxmean,3),round(ds,3))
            print('FWHM:')
            print(round(FWHM,3))
            print('Resolution %')
            print(round(Resolution,3))
            print("Reduced Chi^2")
            print(round(redchisqr,3))

            plt.show()

            #Saves the beautiful plot you just made in the same location as this script
            saveplot = raw_input("Do you wish to save the previous plot? y/n:")
            if saveplot=="y":
                image_name2 = raw_input("save plot as? (filename.png):")
                figure2.savefig(image_name2)

            #A new file is created called calibration_data.csv if one does not already exist in the location as this code
            #If calibration_data.csv already exists, then it adds the data row by row as you agree to save data for calibration.
            #This saves the ADC bin the peak is at (xmean), what the deviation is from the mean (dxmean), and the expected energy the user gave (Energy)
            calibration_data = raw_input("Do you wish to save the pertinent data for calibration? y/n:")
            if calibration_data=="y":
                fields = [xmean, dxmean, Energy]
                with open(r'calibration_data.csv', 'a+') as f:
                    writer = csv.writer(f)
                    writer.writerow(fields)

            repeat_fit = raw_input("Do you wish to repeat the curve fitting? y/n:")
            if repeat_fit == "n":
                print("Have a nice day!")
                break

#The same curve fitting code is repeated over and over in the next couple groups, the only difference is that the curve_fit function is given a different example equation to use to fit.

        else:
            Gaussian_1 = raw_input("Do you wish to fit a linear + Gaussian? y/n:")
            if Gaussian_1 == "y":
                Energy = raw_input("enter known energy for peak of interest in units of keV:")
                x0 = input("enter ADC Bin for peak of interest (from peak find output above):")
                n = input("enter width for region of interest (best guess):")
                itemindex = np.where(np.rint(array(bins[kickass]))==x0)
                P0 = float(maxtab[itemindex[0], 1])
                a0 = 1
                b0 = 1

                pls = []        #pls represents what is going to be an array of bins within the range desired by the user
                hlp = []        #hlp represents what is going to be an array of counts within the range desired by the user
            
                for z in arange(len(bins)):
                    if (bins[z] >= x0 - n/2 and bins[z] <= x0 + n/2):
                        pls.append(bins[z])
                        hlp.append(counts[z])
                        z = z + 1
                    z = z + 1


                x = np.array(pls)
                y = np.array(hlp)
                sigma_Counts = np.sqrt(hlp)
                s0 = np.sqrt(sum((x - x0) ** 2) / (n - 1))

                #Curve Fitting Function
                popt, pcov = curve_fit(firstorderpoly_gauss_function, x, y, p0=[a0, b0, P0, x0, s0], sigma=sigma_Counts)

                # calculate FWHM and resolution % of Gaussian Fit
                da,db, dP, dxmean, ds = [np.sqrt(pcov[j, j]) for j in range(popt.size)]
                a, b, P, xmean, s = popt
                FWHM = 2 * np.sqrt(2 * np.log(2)) * abs(popt[4])
                Resolution = abs(FWHM / popt[3]) * 100

                # Calculating goodness of fit reduced Chi^2
                resids = y - firstorderpoly_gauss_function(x, *popt)
                redchisqr = ((resids / sigma_Counts) ** 2).sum() / float(x.size - 5)

                residplot = max(abs(resids)) + float(0.3 * max(abs(resids)))

                # Plot 1 wide view
                figure3 = plt.figure(file_name,figsize=(10,10))
                gs = gridspec.GridSpec(3, 1, height_ratios=[5, 6, 3])
                axes1 = figure3.add_subplot(gs[0])
                axes1.plot(bins, counts1, color="green", lw=0.5, label="Source Data")
                #axes1.plot(bins, smoothed, color="red", lw=0.5, label="Smoothed Data")
                #axes1.scatter(array(maxtab)[:, 0], array(maxtab)[:, 1], color='black')
                axes1.plot(x, firstorderpoly_gauss_function(x, *popt), lw=2, color="blue", label="Best Fit Curve")
                axes1.set_title((Source, Energy,"keV"), fontsize=20)
                axes1.set_ylabel("Counts", fontsize=15)
                axes1.set_ylim(1, max(firstorderpoly_gauss_function(x,*popt)) + 0.3*float(max(firstorderpoly_gauss_function(x,*popt))))
                axes1.set_xlim(0, 2047)
                #axes1.grid()
                axes1.legend(fontsize=10)

                # Plot 2 Zoomed into Peak
                axes2 = figure3.add_subplot(gs[1])
                axes2.plot(bins, counts1, color="green", lw=0.5)
                # axes2.plot(smoothed, color="red", lw=0.5)
                # axes2.scatter(array(maxtab)[:, 0], array(maxtab)[:, 1], color='black')
                axes2.plot(x, firstorderpoly_gauss_function(x, *popt), lw=2, color="blue")
                axes2.set_ylabel("Counts", fontsize=15)
                axes2.set_ylim(1, max(firstorderpoly_gauss_function(x,*popt)) + 0.3*float(max(firstorderpoly_gauss_function(x,*popt))))
                axes2.set_xlim([popt[3] - (n / 2 + 0.2 * n / 2), popt[3] + (n / 2 + 0.2 * n / 2)])
                #axes2.grid()
                axes2.text(0.1, 0.85, '$g(x) = A+Bx+P\,e^{-(x-xmean)^2/2s^2}$', transform=axes2.transAxes, fontsize=12)
                axes2.text(0.1, 0.75, 'A = {0:0.1f}$\pm${1:0.1f}'.format(a, da), transform=axes2.transAxes,fontsize=10)
                axes2.text(0.1, 0.65, 'B = {0:0.1f}$\pm${1:0.1f}'.format(b, db), transform=axes2.transAxes,fontsize=10)
                axes2.text(0.1, 0.55, 'P = {0:0.1f}$\pm${1:0.1f}'.format(P, dP), transform=axes2.transAxes,fontsize=10)
                axes2.text(0.1, 0.45, 'xmean = {0:0.1f}$\pm${1:0.1f}'.format(xmean, dxmean), transform=axes2.transAxes,fontsize=10)
                axes2.text(0.1, 0.35, 's = {0:0.1f}$\pm${1:0.1f}'.format(s, ds), transform=axes2.transAxes,fontsize=10)
                axes2.text(0.75, 0.85, '$\chi_r^2$ = {0:0.2f}'.format(redchisqr), transform=axes2.transAxes, fontsize=10)
                axes2.text(0.75, 0.75, 'FWHM = {0:0.2f}'.format(FWHM), transform=axes2.transAxes, fontsize=10)
                axes2.text(0.75, 0.65, 'Resolution % = {0:0.2f}'.format(Resolution), transform=axes2.transAxes,fontsize=10)

                # Plot 3 Goodness of Fit
                axes3 = figure3.add_subplot(gs[2])
                axes3.errorbar(x, resids, yerr=sigma_Counts, ecolor="black", fmt="ro")
                axes3.axhline(color="gray", zorder=-1)
                axes3.set_xlabel('ADC Channel', fontsize=15)
                axes3.set_ylabel('residuals', fontsize=15)
                axes3.set_xlim([popt[3] - (n / 2 + 0.2 * n / 2), popt[3] + (n / 2 + 0.2 * n / 2)])
                axes3.set_ylim(-residplot, residplot)

                print("Best Fit Parameters:[A,B,P,mean,std_dev]")
                print(round(a,3),round(b,3),round(P, 3), round(xmean, 3), round(s, 3))
                print("errors in fit parameters:[dA,dB,dP,dxmean,dstd_dev]")
                print(round(da,3),round(db,3),round(dP, 3), round(dxmean, 3), round(ds, 3))
                print('FWHM:')
                print(round(FWHM, 3))
                print('Resolution %')
                print(round(Resolution, 3))
                print("Reduced Chi^2")
                print(round(redchisqr, 3))

                plt.show()

                saveplot = raw_input("Do you wish to save the previous plot? y/n:")
                if saveplot == "y":
                    image_name3 = raw_input("save plot as? (filename.png):")
                    figure3.savefig(image_name3)

                calibration_data = raw_input("Do you wish to save the pertinent data for calibration? y/n:")
                if calibration_data == "y":
                    fields = [xmean, dxmean, Energy]
                    with open(r'calibration_data.csv', 'a+') as f:
                        writer = csv.writer(f)
                        writer.writerow(fields)

                repeat_fit = raw_input("Do you wish to repeat the curve fitting? y/n:")
                if repeat_fit == "n":
                    print("Have a nice day!")
                    break
            else:
                Gaussian_2 = raw_input("Do you wish to fit a quadratic + Gaussian? y/n:")
                if Gaussian_2 == "y":
                    Energy = raw_input("enter known energy for peak of interest in units of keV:")
                    x0 = input("enter ADC Bin for peak of interest (from peak find output above):")
                    n = input("enter width for region of interest (best guess):")
                    itemindex = np.where(np.rint(array(bins[kickass]))==x0)
                    P0 = float(maxtab[itemindex[0], 1])
                    a0 = 1
                    b0 = 1
                    c0 = 1
                    
                    pls = []        #pls represents what is going to be an array of bins within the range desired by the user
                    hlp = []        #hlp represents what is going to be an array of counts within the range desired by the user
            
                    for z in arange(len(bins)):
                        if (bins[z] >= x0 - n/2 and bins[z] <= x0 + n/2):
                            pls.append(bins[z])
                            hlp.append(counts[z])
                            z = z + 1
                        z = z + 1

                    x = np.array(pls)
                    y = np.array(hlp)
                    sigma_Counts = np.sqrt(hlp)
                    s0 = np.sqrt(sum((x - x0) ** 2) / (n - 1))

                    #Curve Fitting function
                    popt, pcov = curve_fit(secondorderpoly_gauss_function, x, y, p0=[a0, b0, c0, P0, x0, s0], sigma=sigma_Counts)

                    # calculate FWHM and resolution % of Gaussian_Fit
                    da, db, dc, dP, dxmean, ds = [np.sqrt(pcov[j, j]) for j in range(popt.size)]
                    a, b, c, P, xmean, s = popt
                    FWHM = 2 * np.sqrt(2 * np.log(2)) * abs(popt[5])
                    Resolution = abs(FWHM / popt[4]) * 100

                    # Calculating goodness of fit reduced Chi^2
                    resids = y - secondorderpoly_gauss_function(x, *popt)
                    redchisqr = ((resids / sigma_Counts) ** 2).sum() / float(x.size - 6)

                    residplot = max(abs(resids)) + float(0.3 * max(abs(resids)))

                    # Plot 1 wide view
                    figure4 = plt.figure(file_name,figsize=(10,10))
                    gs = gridspec.GridSpec(3, 1, height_ratios=[5, 6, 3])
                    axes1 = figure4.add_subplot(gs[0])
                    axes1.plot(bins, counts1, color="green", lw=0.5, label="Source Data")
                    #axes1.plot(bins, smoothed, color="red", lw=0.5, label="Smoothed Data")
                    #axes1.scatter(array(maxtab)[:, 0], array(maxtab)[:, 1], color='black')
                    axes1.plot(x, secondorderpoly_gauss_function(x, *popt), lw=2, color="blue", label="Best Fit Curve")
                    axes1.set_title((Source, Energy,"keV"), fontsize=20)
                    axes1.set_ylabel("Counts", fontsize=15)
                    axes1.set_ylim(1, max(secondorderpoly_gauss_function(x,*popt))+0.3*float(max(secondorderpoly_gauss_function(x,*popt))))
                    axes1.set_xlim(0, 2047)
                    #axes1.grid()
                    axes1.legend(fontsize=10)

                    # Plot 2 Zoomed into Peak
                    axes2 = figure4.add_subplot(gs[1])
                    axes2.plot(bins, counts1, color="green", lw=0.5)
                    # axes2.plot(smoothed, color="red", lw=0.5)
                    # axes2.scatter(array(maxtab)[:, 0], array(maxtab)[:, 1], color='black')
                    axes2.plot(x, secondorderpoly_gauss_function(x, *popt), lw=2, color="blue")
                    axes2.set_ylabel("Counts", fontsize=15)
                    axes2.set_ylim(1, max(secondorderpoly_gauss_function(x,*popt))+0.3*float(max(secondorderpoly_gauss_function(x,*popt))))
                    axes2.set_xlim([popt[4] - (n / 2 + 0.2 * n / 2), popt[4] + (n / 2 + 0.2 * n / 2)])
                    #axes2.grid()
                    axes2.text(0.1, 0.85, '$g(x) = A+Bx+Cx^2+P\,e^{-(x-xmean)^2/2s^2}$', transform=axes2.transAxes,
                               fontsize=12)
                    axes2.text(0.1, 0.75, 'A = {0:0.1f}$\pm${1:0.1f}'.format(a, da), transform=axes2.transAxes,
                               fontsize=10)
                    axes2.text(0.1, 0.65, 'B = {0:0.1f}$\pm${1:0.1f}'.format(b, db), transform=axes2.transAxes,
                               fontsize=10)
                    axes2.text(0.1, 0.55, 'C = {0:0.1f}$\pm${1:0.1f}'.format(c, dc), transform=axes2.transAxes,
                               fontsize=10)
                    axes2.text(0.1, 0.45, 'P = {0:0.1f}$\pm${1:0.1f}'.format(P, dP), transform=axes2.transAxes,
                               fontsize=10)
                    axes2.text(0.1, 0.35, 'xmean = {0:0.1f}$\pm${1:0.1f}'.format(xmean, dxmean),
                               transform=axes2.transAxes, fontsize=10)
                    axes2.text(0.1, 0.25, 's = {0:0.1f}$\pm${1:0.1f}'.format(s, ds), transform=axes2.transAxes,
                               fontsize=10)
                    axes2.text(0.75, 0.85, '$\chi_r^2$ = {0:0.2f}'.format(redchisqr), transform=axes2.transAxes,
                               fontsize=10)
                    axes2.text(0.75, 0.75, 'FWHM = {0:0.2f}'.format(FWHM), transform=axes2.transAxes, fontsize=10)
                    axes2.text(0.75, 0.65, 'Resolution % = {0:0.2f}'.format(Resolution), transform=axes2.transAxes,
                               fontsize=10)

                    # Plot 3 Goodness of Fit
                    axes3 = figure4.add_subplot(gs[2])
                    axes3.errorbar(x, resids, yerr=sigma_Counts, ecolor="black", fmt="ro")
                    axes3.axhline(color="gray", zorder=-1)
                    axes3.set_xlabel('ADC Channel', fontsize=15)
                    axes3.set_ylabel('residuals', fontsize=15)
                    axes3.set_xlim([popt[4] - (n / 2 + 0.2 * n / 2), popt[4] + (n / 2 + 0.2 * n / 2)])
                    axes3.set_ylim(-residplot, residplot)

                    print("Best Fit Parameters:[A,B,C,P,mean,std_dev]")
                    print(round(a, 3), round(b, 3), round(c,3), round(P, 3), round(xmean, 3), round(s, 3))
                    print("errors in fit parameters:[dA,dB,dC,dP,dxmean,dstd_dev]")
                    print(round(da, 3), round(db, 3),round(dc,3), round(dP, 3), round(dxmean, 3), round(ds, 3))
                    print('FWHM:')
                    print(round(FWHM, 3))
                    print('Resolution %')
                    print(round(Resolution, 3))
                    print("Reduced Chi^2")
                    print(round(redchisqr, 3))


                    plt.show()

                    saveplot = raw_input("Do you wish to save the previous plot? y/n:")
                    if saveplot == "y":
                        image_name4 = raw_input("save plot as? (filename.png):")
                        figure4.savefig(image_name4)

                    calibration_data = raw_input("Do you wish to save the pertinent data for calibration? y/n:")
                    if calibration_data == "y":
                        fields = [xmean, dxmean, Energy]
                        with open(r'calibration_data.csv', 'a+') as f:
                            writer = csv.writer(f)
                            writer.writerow(fields)

                    repeat_fit = raw_input("Do you wish to repeat the curve fitting? y/n:")
                    if repeat_fit == "n":
                        print("Have a nice day!")
                        break
                else:
                    Gaussian_3 = raw_input("Do you wish to fit a third order polynomial + Gaussian? y/n:")
                    if Gaussian_3 == "y":
                        Energy = raw_input("enter known energy for peak of interest in units of keV:")
                        x0 = input("enter ADC Bin for peak of interest (from peak find output above):")
                        n = input("enter width for region of interest (best guess):")
                        itemindex = np.where(np.rint(array(bins[kickass]))==x0)
                        P0 = float(maxtab[itemindex[0], 1])
                        a0 = 1
                        b0 = 1
                        c0 = 1
                        d0 = 1
                        
                        pls = []        #pls represents what is going to be an array of bins within the range desired by the user
                        hlp = []        #hlp represents what is going to be an array of counts within the range desired by the user
            
                        for z in arange(len(bins)):
                            if (bins[z] >= x0 - n/2 and bins[z] <= x0 + n/2):
                                pls.append(bins[z])
                                hlp.append(counts[z])
                                z = z + 1
                            z = z + 1

                        x = np.array(pls)
                        y = np.array(hlp)
                        sigma_Counts = np.sqrt(hlp)
                        s0 = np.sqrt(sum((x - x0) ** 2) / (n - 1))

                        #Curve Fitting Function
                        popt, pcov = curve_fit(thirdorderpoly_gauss_function, x, y, p0=[a0, b0, c0, d0, P0, x0, s0],sigma=sigma_Counts)

                        # calculate FWHM and resolution % of Gaussian Fit
                        da, db, dc, dd, dP, dxmean, ds = [np.sqrt(pcov[j, j]) for j in range(popt.size)]
                        a, b, c, d, P, xmean, s = popt
                        FWHM = 2 * np.sqrt(2 * np.log(2)) * abs(popt[6])
                        Resolution = abs(FWHM / popt[5]) * 100

                        # Calculating goodness of fit reduced Chi^2
                        resids = y - thirdorderpoly_gauss_function(x, *popt)
                        redchisqr = ((resids / sigma_Counts) ** 2).sum() / float(x.size - 7)

                        residplot = max(abs(resids)) + float(0.3 * max(abs(resids)))

                        # Plot 1 wide view
                        figure5 = plt.figure(file_name,figsize=(10,10))
                        gs = gridspec.GridSpec(3, 1, height_ratios=[5, 6, 3])
                        axes1 = figure5.add_subplot(gs[0])
                        axes1.plot(bins, counts1, color="green", lw=0.5, label="Source Data")
                        #axes1.plot(bins, smoothed, color="red", lw=0.5, label="Smoothed Data")
                        #axes1.scatter(array(maxtab)[:, 0], array(maxtab)[:, 1], color='black')
                        axes1.plot(x, thirdorderpoly_gauss_function(x, *popt), lw=2, color="blue",
                                   label="Best Fit Curve")
                        axes1.set_title((Source, Energy,"keV"), fontsize=20)
                        axes1.set_ylabel("Counts", fontsize=15)
                        axes1.set_ylim(1, max(thirdorderpoly_gauss_function(x,*popt))+0.3*float(max(thirdorderpoly_gauss_function(x,*popt))))
                        axes1.set_xlim(0, 2047)
                        #axes1.grid()
                        axes1.legend(fontsize=10)

                        # Plot 2 Zoomed into Peak
                        axes2 = figure5.add_subplot(gs[1])
                        axes2.plot(bins, counts1, color="green", lw=0.5)
                        # axes2.plot(smoothed, color="red", lw=0.5)
                        # axes2.scatter(array(maxtab)[:, 0], array(maxtab)[:, 1], color='black')
                        axes2.plot(x, thirdorderpoly_gauss_function(x, *popt), lw=2, color="blue")
                        axes2.set_ylabel("Counts", fontsize=15)
                        axes2.set_ylim(1, max(thirdorderpoly_gauss_function(x,*popt))+0.3*float(max(thirdorderpoly_gauss_function(x,*popt))))
                        axes2.set_xlim([popt[5] - (n / 2 + 0.2 * n / 2), popt[5] + (n / 2 + 0.2 * n / 2)])
                        #axes2.grid()
                        axes2.text(0.1, 0.85, '$g(x) = A+Bx+Cx^2+Dx^3+P\,e^{-(x-xmean)^2/2s^2}$', transform=axes2.transAxes,
                                   fontsize=12)
                        axes2.text(0.1, 0.75, 'A = {0:0.1f}$\pm${1:0.1f}'.format(a, da), transform=axes2.transAxes,
                                   fontsize=10)
                        axes2.text(0.1, 0.65, 'B = {0:0.1f}$\pm${1:0.1f}'.format(b, db), transform=axes2.transAxes,
                                   fontsize=10)
                        axes2.text(0.1, 0.55, 'C = {0:0.1f}$\pm${1:0.1f}'.format(c, dc), transform=axes2.transAxes,
                                   fontsize=10)
                        axes2.text(0.1, 0.45, 'D = {0:0.1f}$\pm${1:0.1f}'.format(d, dd), transform=axes2.transAxes,
                                   fontsize=10)
                        axes2.text(0.1, 0.35, 'P = {0:0.1f}$\pm${1:0.1f}'.format(P, dP), transform=axes2.transAxes,
                                   fontsize=10)
                        axes2.text(0.1, 0.25, 'xmean = {0:0.1f}$\pm${1:0.1f}'.format(xmean, dxmean),
                                   transform=axes2.transAxes, fontsize=10)
                        axes2.text(0.1, 0.15, 's = {0:0.1f}$\pm${1:0.1f}'.format(s, ds), transform=axes2.transAxes,
                                   fontsize=10)
                        axes2.text(0.75, 0.85, '$\chi_r^2$ = {0:0.2f}'.format(redchisqr), transform=axes2.transAxes,
                                   fontsize=10)
                        axes2.text(0.75, 0.75, 'FWHM = {0:0.2f}'.format(FWHM), transform=axes2.transAxes, fontsize=10)
                        axes2.text(0.75, 0.65, 'Resolution % = {0:0.2f}'.format(Resolution), transform=axes2.transAxes,
                                   fontsize=10)

                        # Plot 3 Goodness of Fit
                        axes3 = figure5.add_subplot(gs[2])
                        axes3.errorbar(x, resids, yerr=sigma_Counts, ecolor="black", fmt="ro")
                        axes3.axhline(color="gray", zorder=-1)
                        axes3.set_xlabel('ADC Channel', fontsize=15)
                        axes3.set_ylabel('residuals', fontsize=15)
                        axes3.set_xlim([popt[5] - (n / 2 + 0.2 * n / 2), popt[5] + (n / 2 + 0.2 * n / 2)])
                        axes3.set_ylim(-residplot, residplot)

                        print("Best Fit Parameters:[A,B,C,D,P,mean,std_dev]")
                        print(round(a, 3), round(b, 3), round(c, 3),round(d,3), round(P, 3), round(xmean, 3), round(s, 3))
                        print("errors in fit parameters:[dA,dB,dC,dd,dP,dxmean,dstd_dev]")
                        print(round(da, 3), round(db, 3), round(dc, 3),round(dd,3), round(dP, 3), round(dxmean, 3), round(ds, 3))
                        print('FWHM:')
                        print(round(FWHM, 3))
                        print('Resolution %')
                        print(round(Resolution, 3))
                        print("Reduced Chi^2")
                        print(round(redchisqr, 3))


                        plt.show()

                        saveplot = raw_input("Do you wish to save the previous plot? y/n:")
                        if saveplot == "y":
                            image_name5 = raw_input("save plot as? (filename.png):")
                            figure5.savefig(image_name5)

                        calibration_data = raw_input("Do you wish to save the pertinent data for calibration? y/n:")
                        if calibration_data == "y":
                            fields = [xmean, dxmean, Energy]
                            with open(r'calibration_data.csv', 'a+') as f:
                                writer = csv.writer(f)
                                writer.writerow(fields)

                        repeat_fit = raw_input("Do you wish to repeat the curve fitting? y/n:")
                        if repeat_fit == "n":
                            print("Have a nice day!")
                            break
                    else:

                        repeat_fit = raw_input("There are no other options. Do you wish to repeat the curve fitting? y/n:")
                        if repeat_fit == "n":
                            print("Have a nice day!")
                            break


else:
    print("Have a nice day!")



