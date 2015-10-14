'''
Created on Dec 10, 2014

@author: michal
'''
import main
import numpy as np
import scipy as sp
from scipy import stats
import pandas as pd
import scaling as sc
import matplotlib.pyplot as plt

print " %%% STAT ANALYSIS!  %%% "

# read the session-features matrix from file
fn = '/home/RecSys/matrix_for_ofer'
m = main.read_object_from_file(fn)
#main.main()

# instance of features names (for accessing)
label = main.sLabel()

# calc correlation between two features
print 'correlation: ' , np.corrcoef(m[:,label.AvgTimeSpentItemView], m[:,label.StdTimeSpentItemView])

# calc statistics and percentiles (note! custom percentiles, very useful)
print(pd.DataFrame(m)).describe(percentiles=[.75,.80,.85,.90,.95])

# print values before scaling (a sample of 10 rows)
print m[-10:,label.SessionDuration]
print m[-10:,label.AvgTimeSpentItemView]
print m[-10:,label.StdTimeSpentItemView]


minMaxScaling = []
logScaling1 = []
logScaling2 = []

# append feature indexes that we want to scale 
minMaxScaling.append(label.Week)
logScaling1.append(label.SessionDuration)
logScaling2.append([label.AvgTimeSpentItemView, label.StdTimeSpentItemView])

# apply scaling
sc.minMaxScaling(m, [label.Week])
sc.logTranformating(m, logScaling1, 0.996, 0.004)
sc.logTranformating(m, logScaling2, 0.99, 0.01)

print 'correlation: ' , np.corrcoef(m[:,label.AvgTimeSpentItemView], m[:,label.StdTimeSpentItemView])

# print values AFTER scaling (a sample of 10 rows)
print m[-10:,label.SessionDuration]
print m[-10:,label.AvgTimeSpentItemView]
print m[-10:,label.StdTimeSpentItemView]

# plot the feature distribution 
x=m[:,label.SessionDuration]
#plt.plot(x)
plt.hist(x, bins=10)#, normed=True, log=True)
plt.show()


# print statistics after scaling
print(pd.DataFrame(m)).describe(percentiles=[.75,.80,.85,.90,.95])
