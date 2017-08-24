# -*- coding: utf-8 -*-
"""
Created on Thu Aug 10 11:29:19 2017

@author: Cameron
"""


import matplotlib.pyplot as plt
import numpy as np
import sys
from numpy import NaN, Inf, arange, isscalar, asarray, array
from matplotlib.pyplot import plot, scatter, show
from scipy.optimize import curve_fit
import math
import matplotlib.gridspec as gridspec
from math import factorial

#Defining the necessary functions

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
      """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
   3     The Savitzky-Golay filter removes high frequency noise from data.
   4     It has the advantage of preserving the original shape and
   5     features of the signal better than other types of filtering
   6     approaches, such as moving averages techniques.
   7     Parameters
   8     ----------
   9     y : array_like, shape (N,)
  10         the values of the time history of the signal.
  11     window_size : int
  12         the length of the window. Must be an odd integer number.
  13     order : int
  14         the order of the polynomial used in the filtering.
  15         Must be less then `window_size` - 1.
  16     deriv: int
  17         the order of the derivative to compute (default = 0 means only smoothing)
  18     Returns
  19     -------
  20     ys : ndarray, shape (N)
  21         the smoothed signal (or it's n-th derivative).
  22     Notes
  23     -----
  24     The Savitzky-Golay is a type of low-pass filter, particularly
  25     suited for smoothing noisy data. The main idea behind this
  26     approach is to make for each point a least-square fit with a
  27     polynomial of high order over a odd-sized window centered at
  28     the point.
  29     Examples
  30     --------
  31     t = np.linspace(-4, 4, 500)
  32     y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
  33     ysg = savitzky_golay(y, window_size=31, order=4)
  34     import matplotlib.pyplot as plt
  35     plt.plot(t, y, label='Noisy signal')
  36     plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
  37     plt.plot(t, ysg, 'r', label='Filtered signal')
  38     plt.legend()
  39     plt.show()
  40     References
  41     ----------
  42     .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
  43        Data by Simplified Least Squares Procedures. Analytical
  44        Chemistry, 1964, 36 (8), pp 1627-1639.
  45     .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
  46        W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
  47        Cambridge University Press ISBN-13: 9780521880688
  48     """
     

   
  
      try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
      except ValueError, msg:
          raise ValueError("window_size and order have to be of type int")
      if window_size % 2 != 1 or window_size < 1:
          raise TypeError("window_size size must be a positive odd number")
      if window_size < order + 2:
          raise TypeError("window_size is too small for the polynomials order")
      order_range = range(order+1)
      half_window = (window_size -1) // 2
      # precompute coefficients
      b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
      m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
      # pad the signal at the extremes with
      # values taken from the signal itself
      firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
      lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
      y = np.concatenate((firstvals, y, lastvals))
      return np.convolve( m[::-1], y, mode='valid')

#def smoothListGaussian(myarray,degree=5):
#smooth_list_Gaussian smooths a data set by averaging using a weigthed Gaussian
#Returns a smoothed 1D array of identical size of input data used to better identify peaks.


    #myarray = np.pad(myarray, (degree-1,degree-1), mode='edge')

    #window=degree*2-1

    
    #weight=np.arange(-degree+1, degree)/window

    #weight = np.exp(-(16*weight*2))

    #weight /= sum(weight)

    #smoothed = np.convolve(myarray, weight, mode='valid')

    #return smoothed

    

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
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
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

file_name = raw_input("Please input Source data: ")       #prompts user to input a file name, must be in .csv format

V1 = []                                           #initializing the lists for each channel to be filled from file
W1 = []
with open(file_name) as csvfile:                       #appends each column of data into a 1D numpy array.
    readCSV = csv.reader(csvfile, delimiter=',')

    for row in readCSV:
        V1.append(float(row[0]))
        W1.append(float(row[1]))
Channel = np.array(V1)
Counts1 = np.array(W1)
#Smoothing data for peak finding using the Gaussian averaging function defined above

#smoothed = smoothListGaussian(Counts1,degree=5)
smoothed= savitzky_golay(Counts1, 5, 3)

#naming radiation source
Source = raw_input("Enter name of Radiation Source:")


# Finding Peaks of the 'smoothed' data set using the peakdet function defined above
Delta = input("enter minimum height of expected peaks (required):")
series = smoothed
maxtab, mintab = peakdet(series, Delta)

CountRange = max(Counts1)+100
ChannelMin = 0
ChannelMax = 2047

#Initial Plot of data, smoothed data, and estimated peaks
figure1 = plt.figure(file_name, figsize=(10,10))
axes1 = plt.subplot(111)
axes1.plot(Counts1, color="green", lw=0.5, label="Raw")
axes1.plot(smoothed, color="red", lw=1, label="Smoothed")
axes1.scatter(array(maxtab)[:, 0], array(maxtab)[:, 1], color='black')
axes1.set_title(Source, fontsize=20)
axes1.set_xlabel("ADC Values", fontsize=20)
axes1.set_ylabel("Counts", fontsize=10)
axes1.set_ylim([1, CountRange])
axes1.set_xlim([ChannelMin, ChannelMax])
axes1.grid()
axes1.legend()

print("Peaks at:")
print("[ADC Bin,Count]")
print(maxtab)
plt.show()

Manual_Params = raw_input("Do you wish to manually re-enter the plotting parameters? y/n:")
if Manual_Params=="y":
    while True:
        CountRange = input("Enter Maximum Counts value:")
        ChannelMin = input("Enter minimum ADC Channel:")
        ChannelMax = input("Enter maximum ADC Channel:")
        Delta = input("enter minimum height of expected peaks (required):")
        windows=input("Enter windows width:")
        smoothed= savitzky_golay(Counts1, windows, 3)
        series = smoothed
        maxtab, mintab = peakdet(series, Delta)


        # Manually entered plot parameters to look at a specific peak
        figure1 = plt.figure(file_name)
        axes1 = plt.subplot(111)
        axes1.plot(Counts1, color="green", lw=0.5, label="Raw")
        axes1.plot(smoothed, color="red", lw=1, label="Smoothed")
        axes1.scatter(array(maxtab)[:, 0], array(maxtab)[:, 1], color='black')
        axes1.set_title(Source, fontsize=20)
        axes1.set_xlabel("ADC Values", fontsize=20)
        axes1.set_ylabel("Counts", fontsize=20)
        axes1.set_ylim([1, CountRange])
        axes1.set_xlim([ChannelMin, ChannelMax])
        axes1.grid()
        axes1.legend()
        print("Peaks at:")
        print("[ADC Bin,Count]")
        print(maxtab)

        

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
            itemindex = np.where(maxtab==x0)
            P0 = float(maxtab[itemindex[0],1])

            x = Channel[x0 - n/2:x0 + n/2]
            y = Counts1[x0 - n/2:x0 + n/2]
            sigma_Counts = np.sqrt(Counts1[x0 - n/2:x0 + n/2])
            s0 = np.sqrt(sum((x-x0)**2)/(n-1))

            #Curve Fitting Function
            popt, pcov = curve_fit(gauss_function, x, y, p0=[P0, x0, s0],sigma=sigma_Counts)


            # calculate FWHM and resolution % of Gaussian Fit
            dP, dxmean, ds = [np.sqrt(pcov[j, j]) for j in range(popt.size)]
            P, xmean, s = popt
            FWHM = 2 * np.sqrt(2 * np.log(2)) * abs(popt[2])
            Resolution = abs(FWHM / popt[1]) * 100

            # Calculating goodness of fit reduced Chi^2
            resids = y - gauss_function(x, *popt)
            redchisqr = ((resids / sigma_Counts) ** 2).sum() / float(x.size - 3)
            new_resids=resids
            for i in np.arange(len(resids)):
                new_resids[i]=resids[i]/y[i]
                
            

            residplot = max(abs(resids))+float(0.3*max(abs(resids)))


            #Plot 1 wide view
            figure2 = plt.figure(file_name,figsize=(10,10))
            gs = gridspec.GridSpec(3, 1, height_ratios=[5, 6, 3])
            axes1 = figure2.add_subplot(gs[0])
            axes1.plot(Counts1, color="green", lw=0.5, label="Source Data")
            axes1.plot(smoothed, color="red", lw=0.5, label="Smoothed Data")
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
            axes2.plot(Counts1, color="green", lw=0.5)
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

            saveplot = raw_input("Do you wish to save the previous plot? y/n:")
            if saveplot=="y":
                image_name2 = raw_input("save plot as? (filename.png):")
                figure2.savefig(image_name2)

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
        else:
            Gaussian_1 = raw_input("Do you wish to fit a linear + Gaussian? y/n:")
            if Gaussian_1 == "y":
                Energy = raw_input("enter known energy for peak of interest in units of keV:")
                x0 = input("enter ADC Bin for peak of interest (from peak find output above):")
                n = input("enter width for region of interest (best guess):")
                itemindex = np.where(maxtab == x0)
                P0 = float(maxtab[itemindex[0], 1])
                a0 = 1
                b0 = 1



                x = Channel[x0 - n/2:x0 + n/2]
                y = Counts1[x0 - n/2:x0 + n/2]
                sigma_Counts = np.sqrt(Counts1[x0 - n/2:x0 + n/2])
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
                axes1.plot(Counts1, color="green", lw=0.5, label="Source Data")
                axes1.plot(smoothed, color="red", lw=0.5, label="Smoothed Data")
                # axes1.scatter(array(maxtab)[:, 0], array(maxtab)[:, 1], color='black')
                axes1.plot(x, firstorderpoly_gauss_function(x, *popt), lw=2, color="blue", label="Best Fit Curve")
                axes1.set_title((Source, Energy,"keV"), fontsize=20)
                axes1.set_ylabel("Counts", fontsize=15)
                axes1.set_ylim(1, max(firstorderpoly_gauss_function(x,*popt)) + 0.3*float(max(firstorderpoly_gauss_function(x,*popt))))
                axes1.set_xlim(0, 2047)
                #axes1.grid()
                axes1.legend(fontsize=10)

                # Plot 2 Zoomed into Peak
                axes2 = figure3.add_subplot(gs[1])
                axes2.plot(Counts1, color="green", lw=0.5)
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
                    itemindex = np.where(maxtab == x0)
                    P0 = float(maxtab[itemindex[0], 1])
                    a0 = 1
                    b0 = 1
                    c0 = 1

                    x = Channel[x0 - n/2:x0 + n/2]
                    y = Counts1[x0 - n/2:x0 + n/2]
                    sigma_Counts = np.sqrt(Counts1[x0 - n/2:x0 + n/2])
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
                    axes1.plot(Counts1, color="green", lw=0.5, label="Source Data")
                    axes1.plot(smoothed, color="red", lw=0.5, label="Smoothed Data")
                    # axes1.scatter(array(maxtab)[:, 0], array(maxtab)[:, 1], color='black')
                    axes1.plot(x, secondorderpoly_gauss_function(x, *popt), lw=2, color="blue", label="Best Fit Curve")
                    axes1.set_title((Source, Energy,"keV"), fontsize=20)
                    axes1.set_ylabel("Counts", fontsize=15)
                    axes1.set_ylim(1, max(secondorderpoly_gauss_function(x,*popt))+0.3*float(max(secondorderpoly_gauss_function(x,*popt))))
                    axes1.set_xlim(0, 2047)
                    #axes1.grid()
                    axes1.legend(fontsize=10)

                    # Plot 2 Zoomed into Peak
                    axes2 = figure4.add_subplot(gs[1])
                    axes2.plot(Counts1, color="green", lw=0.5)
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
                        itemindex = np.where(maxtab == x0)
                        P0 = float(maxtab[itemindex[0], 1])
                        a0 = 1
                        b0 = 1
                        c0 = 1
                        d0 = 1

                        x = Channel[x0 - n/2:x0 + n/2]
                        y = Counts1[x0 - n/2:x0 + n/2]
                        sigma_Counts = np.sqrt(Counts1[x0 - n/2:x0 + n/2])
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
                        axes1.plot(Counts1, color="green", lw=0.5, label="Source Data")
                        axes1.plot(smoothed, color="red", lw=0.5, label="Smoothed Data")
                        # axes1.scatter(array(maxtab)[:, 0], array(maxtab)[:, 1], color='black')
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
                        axes2.plot(Counts1, color="green", lw=0.5)
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



