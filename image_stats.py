# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 08:52:52 2022

@author: sharrm
"""
import matplotlib
import matplotlib.pyplot as plt
import gdal
import numpy as np
from scipy import stats


april = r"P:\SDB\Florida Keys\Final_Projects\bathy_20210425.tif"
may = r"P:\SDB\Florida Keys\Final_Projects\bathy_20210505.tif"

april = gdal.Open(r"P:\SDB\Florida Keys\Final_Projects\bathy_20210425.tif")
may = gdal.Open(r"P:\SDB\Florida Keys\Final_Projects\bathy_20210505.tif")

april_array = april.GetRasterBand(1).ReadAsArray()
may_array = may.GetRasterBand(1).ReadAsArray()

np_april = april_array.flatten()
np_may = may_array.flatten()

np.size(np_april)
np.size(np_may)

x_april = np_april[~np.isnan(np_april)]
x_may = np_may[~np.isnan(np_may)]

np.size(x_april)
np.size(x_may)

print("April Mean:", np.mean(x_april)) #7.439085
print("April Variance", np.var(x_april)) #1.8656594
print("May Mean", np.mean(x_may)) #6.559908
print("May Variance", np.var(x_may)) #4.6548767

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html

t, p = stats.ttest_ind(x_april, x_may, equal_var='FALSE')
print("t-stat: %.3f, p-value: %.3f" % (t, p)) #552.7091298685874 0.0

# https://matplotlib.org/stable/gallery/color/named_colors.html

fig1, ax1 = plt.subplots(figsize=(8, 6), dpi=80)
result = plt.hist(x_april, bins=40, color='lightsteelblue', edgecolor='k', alpha=0.65)
plt.axvline(x_april.mean(), color='darkblue', linestyle='dashed', linewidth=1, label='mean')
ax1.set_title("Distribution of SDB Depths (25April2021)")
ax1.set_ylabel("Count (thousands)"), ax1.set_xlabel("Depth (m)")
ax1.legend(loc='upper right')
ax1.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, p: format(int(y/1000))))
ax1.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x))))
plt.show()

fig2, ax2 = plt.subplots(figsize=(8, 6), dpi=80)
result = plt.hist(x_may, bins=40, color='powderblue', edgecolor='k', alpha=0.65)
plt.axvline(x_may.mean(), color='darkcyan', linestyle='dashed', linewidth=1, label='mean')
ax2.set_title("Distribution of SDB Depths (05May2021)")
ax2.set_ylabel("Count (thousands)"), ax2.set_xlabel("Depth (m)")
ax2.legend(loc='upper right')
ax2.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, p: format(int(y/1000))))
ax2.get_xaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x))))
plt.show()

# https://matplotlib.org/stable/gallery/pyplots/boxplot_demo_pyplot.html

fig3, ax3 = plt.subplots(figsize=(8, 6), dpi=80)
fliers = dict(marker='.', markerfacecolor='lightcoral', markersize=3, linestyle='none', markeredgecolor='none')
medianprops = dict(color="steelblue",linewidth=1.25)
plt.boxplot([x_april, x_may], flierprops=fliers, patch_artist=True, boxprops=dict(facecolor='white'), medianprops=medianprops) #showfliers=Fals
ax3.set_xticklabels(['April', 'May'])
ax3.set_title("Distribution of SDB Depths")
ax3.set_ylabel("Depth (m)")
plt.show()
