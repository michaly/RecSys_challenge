'''
Created on Dec 31, 2014

@author: nancy
'''

import os
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
import dateutil.parser as dateparser
from scipy.sparse import coo_matrix
import operator
#import graphlab as gl
import sklearn.metrics as metrics
from sklearn.utils import resample
from sklearn import svm, cross_validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

'''Determining the pathaway for the data and the files we save in order to decrease the running time'''
workingDirectory = os.getcwd()
clickFileName = ''.join([workingDirectory,'\\yoochoose-clicks-sample2.dat']) #1M
buyFileName = workingDirectory + '\\yoochoose-buys.dat'
testFileName = workingDirectory + '\\yoochoose-test.dat'#yoochoose-test.dat'
sessionsFeaturesFileTrain = workingDirectory + '\\sessionsFeaturesTrain'
sessionsFeaturesFileTest = workingDirectory + '\\sessionsFeaturesTest'
itemsdictFileTrain = workingDirectory + '\\itemsdictTrain'
itemsdictFileTest = workingDirectory + '\\itemsdictTest'
timespentdictFileTest= workingDirectory + '\\timespentdictTest'
modelFile = workingDirectory + '\\model'
submissionFile = workingDirectory + '\\solution.dat'


'''Defining global dictionaries to store items or users based on a specific feature'''
global sessionsFeatures; sessionsFeatures = defaultdict(list)
global timeSpentList; timeSpentList = defaultdict(list)
global categorydict; categorydict = {}
global itemsdict; itemsdict={}
global itemsClickDict; itemsClickDict ={}
global timeSpentItemDict; timeSpentItemDict={}
global timeSpentItemDictList; timeSpentItemDictList={}

##%%
# Change the working directory to the location of "main.py":
#abspath = os.path.abspath(__file__)
#dname = os.path.dirname(abspath)
#os.chdir(dname)
##%%

class DateHandler:
'''dealing with the given time format'''
    def __init__(self, ts):
        
        self.fullDate = self.formatDate(ts)
        self.time = self.getTime()
        
    def formatDate(self,ts):
     
        """ 
        Translate a ISO 8601 datetime string into a Python datetime object
        
        Parameters
        ----------
        d : str
            ISO 8601 datetime string - 'YYYY-mm-DDTHH:MM:SS.SSSSSSZ'
        
        Returns
        -------
        dateatime
            
        """
        return dateparser.parse(ts)

    def getWeekNumber(self):
        return self.fullDate.isocalendar()[1]
    
    def getDayInWeek(self):
        '''
        
        '''
        return self.fullDate.strftime("%A")
        
    def getTime(self):
        ''' Return full Time as string'''
        return datetime.time(self.fullDate)
    
    def getPartOfDay(self):
        ''' 
        Get time as string
        Returns Part of the Day as string
        '''
        t = str(self.time)
        
        if (t >= '05:00:00.000000' and t <= '11:00:00.000000') : return 'Morning'
        if (t >= '12:00:00.000000' and t <= '16:00:00.000000') : return 'Noon'
        if (t >= '17:00:00.000000' and t <= '21:00:00.000000') : return 'Evening'
        else: return 'Night'
    
    def getHour(self):
        ''' Return hour 1:00 - 00:00 as string'''
        return str(self.fullDate.hour)
   
class sLabel():
'''the class saves the sessions features after extraction- this class is for coding purposes only '''
    TotalTrx                    =   0
    Week                        =   1
    
    isSunday                    =   2
    isMonday                    =   3
    isTuesday                   =   4
    isWednesday                 =   5 
    isThursday                  =   6 
    isFriday                    =   7 
    isSaturday                  =   8
         
    Hour                        =   9
    
    isMorning                   =   10
    isNoon                      =   11
    isEvening                   =   12
    isNight                     =   13
    
    PrimarySubjectInterest      =   14
    SecondarySubjectInterest    =   15
    SessionDuration             =   16
    NumberDistinctCategories    =   17
    AvgTimeSpentItemView        =   18
    StdTimeSpentItemView        =   19
    PctDoubleClicks             =   20  # (total num of trx / distinct items in session ) = (average trx for item, in session)
    MaxDoubleClicks             =   21  # (number of clicks for the item in the session has been clicked the most)
    IsBought                    =   22     # target variable 
    

def loadData(fileName, delimiter, dtype):
    '''returns the stored data file '''
    print("START -----> loadData " + str(datetime.now()))
    
    data = np.loadtxt(fileName, delimiter=delimiter, dtype=dtype)
    
    print("END -----> loadData " + str(datetime.now()))
    return data

def findBoughtSessions(click, buy):
'''crossing the datasets of clicks and buys in order to get the sessions which had buying event'''
    boughtSession = np.intersect1d(click['sesId'], buy['sesId'], assume_unique=False)
    
    return boughtSession
   
def extractFeatures(data):
    
    """
    Feature generation from the raw click data
    
    Parameters
        ----------
        data : numpy array
             (list of lists, each record is a list)
        
        Returns
        -------
        
    """
    print("START -----> extractFeatures " + str(datetime.now()))
    
    prevSesId = 0
    global timeSpentList; 
    # run on each record on the click file
    for trx in data:
        
        sId = trx['sesId']
        date = trx['TS']
        sCategory = trx['category']
        sItem = trx['itemId']
        
        
        dateHandler = DateHandler(date)
        date = dateHandler.fullDate
        
        # Actions to do in case of a new session only
        if not sId in sessionsFeatures:
            initNewSession(sId)
            setDateFeatures(sId, dateHandler)
            prevDate = date
            itemsdict[sId]={}
            itemsdict[sId][sItem]=1
            categorydict[sId]={}
            categorydict[sId][sCategory]=1
        else:
            if not sItem in itemsdict[sId]:
                itemsdict[sId][sItem] =1
            else:
                itemsdict[sId][sItem]+=1   
            

        # Actions to do for each transaction in session 
        increaseTimeSpent(sId, date, prevDate, sItem)
        increaseTrxCounterForSession(sId)
        prevDate = date
    
    calcAggTimeSpent()
    calcCountCategoryForSession()
    calcPctDoubleClicks()
    calcMaxDoubleClicks()
    calcItemsClickList()
    calcPctTimeSpentList()
    
    print("END -----> extractFeatures " + str(datetime.now()))
            
    return

def increaseTimeSpent(sId, date, prevDate, sItem):
    '''used by ExtractFeatures the function sums the time of each session for items'''
    global timeSpentList
    timeSpent = date - prevDate
    timeSpent = timeSpent.total_seconds()
    timeSpentList[sId].append(timeSpent)
    if not sId in timeSpentItemDict:
        timeSpentItemDict[sId] = {}
    if not sItem in  timeSpentItemDict[sId]:
        timeSpentItemDict[sId][sItem] = timeSpent
    else:
         timeSpentItemDict[sId][sItem] +=timeSpent
    return 
    
def calcAggTimeSpent():
    '''calculates the sum of time spent and places the values in the assigned features'''
    global sessionsFeatures
    global timeSpentList
    print("START -----> calcAggTimeSpent " + str(datetime.now()))

    for sId in sessionsFeatures:
        l = timeSpentList[sId]        
        if (len(l) > 0):
            avg = np.mean(l, axis=0) #sum(timeSpentList) / float(len(timeSpentList))
            std = np.std(l, axis=0)
            sessionsFeatures[sId][sLabel.AvgTimeSpentItemView] = avg 
            sessionsFeatures[sId][sLabel.StdTimeSpentItemView] = std   
            sessionsFeatures[sId][sLabel.SessionDuration] = sum(l)
 
    print("END -----> calcAggTimeSpent " + str(datetime.now()))
    return

def initNewSession(sId):
        
    """
    Initialize container with zeros
    """
    global sessionsFeatures
    
     
    for i in [attr for attr in dir(sLabel) if not callable(attr) and not attr.startswith("__")]:
        sessionsFeatures[sId].append(0)
    
    return
            
def increaseTrxCounterForSession(sId):
    """
    Counts the number of transaction in session
    
    Parameters
        ----------
        sId : str
            the session unique id
        
    Returns
    -------
    None
    """
    global sessionsFeatures
    sessionsFeatures[sId][sLabel.TotalTrx] += 1
    return
    
def calcCountCategoryForSession():
    '''calculates the number of distinct categories in the sessionFeatures dict for session ID'''
    for sId in sessionsFeatures:
        sessionsFeatures[sId][sLabel.NumberDistinctCategories] = len(categorydict[sId].keys())
    return
    
def calcPctDoubleClicks():
    '''calculates the number of distinct items in session / total num of trx for session'''
    for sId in sessionsFeatures:
        sessionsFeatures[sId][sLabel.PctDoubleClicks] = 1/(float(len(itemsdict[sId].keys()))/float(sessionsFeatures[sId][sLabel.TotalTrx]))
    return

def calcMaxDoubleClicks():
    '''(number of clicks for the item in the session has been clicked the most)'''
    for sId in sessionsFeatures:
        sessionsFeatures[sId][sLabel.MaxDoubleClicks] = max(itemsdict[sId].values())
    return
    
def calcItemsClickList():
    '''calculates for all sesID sorted list containing (itemID, NumClicks)'''
    global itemsClickDict
    for sId in itemsdict:
        itemsClickDict[sId]=[]
        for itemId in itemsdict[sId]:
            itemsClickDict[sId].append((itemId,itemsdict[sId][itemId]))
    for sId in itemsClickDict:
        itemsClickDict[sId] = sorted(itemsClickDict[sId], key = lambda k:k[1], reverse= True)

    return

def calcPctTimeSpentList():
    '''calculates for all sesID sorted list containing (itemID, timeSpent/SessionDuration)'''
    for sId in timeSpentItemDict:
        timeSpentItemDictList[sId]=[]
        for itemId in timeSpentItemDict[sId]:
            if not sessionsFeatures[sId][sLabel.SessionDuration] == 0.0:
                timeSpentItemDictList[sId].append((itemId,timeSpentItemDict[sId][itemId]/sessionsFeatures[sId][sLabel.SessionDuration]))
    for sId in timeSpentItemDictList:
        timeSpentItemDictList[sId]= sorted(timeSpentItemDictList[sId], key = lambda l:l[1], reverse=True)
        
    return      
def setDateFeatures(sId, d):
    """ 
    Update sessionsFeatures with all date features only in the first transaction (we take the data of the first action) 
    
    Parameters
    ----------
    d : dateHandler  
        class object
    
    Returns
    -------
    None
        
    """
    
    sessionsFeatures[sId][sLabel.Week] = int(d.getWeekNumber())
    day = d.getDayInWeek()
    sessionsFeatures[sId][sLabel.Hour] = int(d.getHour())
    partofDay = d.getPartOfDay()
    updateCategoricalDateFeatures(sId, day, partofDay)
    
    return 

def updateCategoricalDateFeatures(sId, d, pod):
    
    """
    Transform date categorical features to binary.
    Parameters
    -----------
    sId - str
            session id
    d - str
            day of week
    pod - str
            part of day
    Returns
    --------
    None
    """
    
    global sessionsFeatures
    
    if (d=='Sunday'):
        sessionsFeatures[sId][sLabel.isSunday] = 1
    elif (d=='Monday'):
        sessionsFeatures[sId][sLabel.isMonday] = 1
    elif (d=='Tuesday'):
        sessionsFeatures[sId][sLabel.isTuesday] = 1
    elif (d=='Wednesday'):
        sessionsFeatures[sId][sLabel.isWednesday] = 1
    elif (d=='Thursday'):
        sessionsFeatures[sId][sLabel.isThursday] = 1
    elif (d=='Friday'):
        sessionsFeatures[sId][sLabel.isFriday] = 1
    else: #Saturday
        sessionsFeatures[sId][sLabel.isSaturday] = 1
        
         
    if (pod=='Morning'):
        sessionsFeatures[sId][sLabel.isMorning] = 1
    elif (pod=='Noon'):
        sessionsFeatures[sId][sLabel.isNoon] = 1
    elif (pod=='Evening'):
        sessionsFeatures[sId][sLabel.isEvening] = 1
    else: #Night
        sessionsFeatures[sId][sLabel.isNight] = 1
    return

def createLabelForClassification(click, buy):
    '''determines whether the session in the sessionsFeatures dataset will buy any item based on bought list'''
    print("START -----> createLabelForClassification " + str(datetime.now()))
    
    global sessionsFeatures
    
    bought = findBoughtSessions(click, buy)
    
    for sId in sessionsFeatures:
        if sId in bought:
            sessionsFeatures[sId][sLabel.IsBought] = 1
            
    print("END -----> createLabelForClassification " + str(datetime.now()))
    
    return
  
def cross_validation(clf, sample_data, target):
    '''applying cross validation classifier and score on a dataset'''
    print("START ... cross_validation" + str(datetime.now()))
    
    # clf is a classification object from sklearn algorithms
    strat_cv = cross_validation.StratifiedKFold(target, n_folds=5)
    scores = cross_validation.cross_val_score(clf, sample_data, target, cv=strat_cv, scoring='precision')
    print scores
    
    print("END ... cross_validation" + str(datetime.now()))
    
    return

def predict(data):
    '''applying classifier on the data, prints a report based on the classifier whether will be bought and returns the list of bought sessions'''
    print("Start Predict ... " + str(datetime.now()))
    clf = read_object_from_file(modelFile)
    print clf
    
    labels_pred = clf.predict(data)
    print 'Number of bought users: ',sum(labels_pred)
    bought = list(np.where(labels_pred == 1)[0])
    
    print("End Predict ... " + str(datetime.now()))
    
    return bought

def classifySessions(data):
    '''choosing the right classification method and applying it on the data '''
    print("Start classifySessions " + str(datetime.now()))
    
    # Prepare data for learning
    labels_true = data[:,-1]
    not_bought = data[np.where(data[:,sLabel.IsBought] == 0.0)]
    bought = data[np.where( data[:,sLabel.IsBought] == 1.0 )] 
    dist_ratio = 1 # The required ratio between bought and not bought (for example, if 10 then the ratio is 10:1- bought:not bought)
    not_bought = resample(not_bought, n_samples=len(bought)*dist_ratio, random_state=1)    # Under sampling 
    sample_data = np.concatenate([not_bought,bought])
    target = sample_data[:,-1]
    # Remove the target variable 
    sample_data = sample_data[:,0:-1]
    data = data[:,0:-1]
    
    
    ## SVM
    #kernel='rbf' #kernel='linear'
    #clf = svm.SVC(kernel=kernel, C=1)
    
    ## Decision Tree
    #clf = DecisionTreeClassifier(random_state=0)
    
    # KNN
    #clf = KNeighborsClassifier(n_neighbors=50)
    
    # Random Forest
    #clf = RandomForestClassifier(n_estimators=10)
    
    # Choosing the parameters of the model
    #cross_validation(clf, sample_data, target)
    
    clf = GaussianNB()
    
    print("FIT ... " + str(datetime.now()))
    clf.fit(sample_data, target)
    write_object_to_file(clf,modelFile)
    print("Predict ... " + str(datetime.now()))
    labels_pred = clf.predict(sample_data)
    print("Classification report ... " + str(datetime.now()))
    print(metrics.classification_report(target, labels_pred, target_names=['class 0', 'class 1']))
    
    print("End classifySessions " + str(datetime.now()))
    return 
    
def TimeSpentRecommender(buyers):
    '''uses a simplified method to recommend items to buy based on items the session spent most of his time in the session'''
    print("Start Recommend Sessions " + str(datetime.now()))
    d = defaultdict(list)
    
    for sId in buyers:
        if timeSpentItemDictList[sId]:
            for i in range(5):
                if len(timeSpentItemDictList[sId]) > i:
                    if timeSpentItemDictList[sId][i][1]>=0.01:
                        d[sId].append(timeSpentItemDictList[sId][i][0])

    print("Starting to write file for submission... " + str(datetime.now()))    
    writeForSubmission(submissionFile,d)    
    
    print("End Recommend Sessions " + str(datetime.now()))
    return

################## Recommender methods that gave worse results than TimeSpentRecommender: ##################

def recommendSessions(buyers):
    '''uses Graphlab package (ranking factorization) to recommend items which viewed in sessions as well as new items(weren't viewed during the session)'''
    print("Start Recommend Sessions " + str(datetime.now()))
    # Build  Session Items incidence matrix
    data = matrixBuilder()
    
    model = gl.ranking_factorization_recommender.create(data, user_id="session", item_id="item", target="frequency")
    recs = model.recommend(users=buyers, exclude_known=False, k=1) 
    
    d = defaultdict(list) 
    for rec in recs:
        sId=rec["session"]
        iId=rec["item"]
        d[sId].append(iId)
    
    print("Starting to write file for submission... " + str(datetime.now()))    
    writeForSubmission(submissionFile,d)
    
    return

def naiveReccomender(buyers,k):
    '''uses a simplified method to recommend items to buy based on the most viewed items in the session'''
    print("Start Recommend Sessions " + str(datetime.now()))
    
    d = defaultdict(list) 
    

     # Find for each session, the item that was viewed the most during this session 
    for ses in buyers:
         items = itemsdict[ses]
         freq_item = dict(sorted(items.iteritems(), key=operator.itemgetter(1), reverse=True)[:k]).keys()
         d[ses].extend(freq_item)


    print("Starting to write file for submission... " + str(datetime.now()))    
    writeForSubmission(submissionFile,d)    
    
    print("End Recommend Sessions " + str(datetime.now()))
    return
    
##########################################################################################
    
def writeForSubmission(filePath,d):
    '''an auxiliary function to write solution file for submission'''
    target = open(filePath, 'w') # Open for writing.  The file is created if it does not exist.
    
    for key, values in d.iteritems():
        target.write(str(key) + ";")
        for idx,val in enumerate(values):
            if(idx):
                target.write("," + str(val)  )
            else:
                target.write(str(val))
        target.write("\n")
          
    target.close()
    return

def write_object_to_file(obj,fn):
    '''an auxiliary function to load dataset for further use and decrease running time in future runs'''
    print("START write_object_to_file " + str(datetime.now()))
    import pickle
    out = open(fn, 'w')
    pickle.dump(obj, out)
    out.close()
    return

def read_object_from_file(fn):
    '''an auxiliary function to load dataset for further use and decrease running time in future runs'''
    print("START -----> read_object_from_file " + str(datetime.now()))
    
    import pickle
    o = open(fn)
    x = pickle.load(o)
    o.close()
    
    print("END -----> read_object_from_file " + str(datetime.now()))
     
    return x
    
def dataPreparation(m):
    '''using the scaling class in the scaling file, the function countify features to fit on the same scale'''
    import scaling as sc
    minMaxScaling = []
    logScaling1 = []
    logScaling2 = []
    logScaling3=[]
    logScaling4=[]
    
    # append feature indexes that we want to scale 
    minMaxScaling.append(sLabel.Week)
    logScaling1.append(sLabel.SessionDuration)
    logScaling2.append([sLabel.AvgTimeSpentItemView, sLabel.StdTimeSpentItemView])
    logScaling3.append([sLabel.MaxDoubleClicks,sLabel.PctDoubleClicks])
    logScaling4.append(sLabel.TotalTrx)
    
    # apply scaling
    sc.minMaxScaling(m, minMaxScaling)
    sc.logTranformating(m, logScaling1, 0.996, 0.004)
    sc.logTranformating(m, logScaling2, 0.99, 0.01)
    sc.logTranformating(m, logScaling3, 0.5, 0.5)
    sc.logTranformating(m, logScaling4, 0.8, 0.2)
    m[:,sLabel.Hour] /= 23.0
    
    return m

def matrixBuilder():
    ''''returns a matrix of sessions  and items. for each sessions-item value whether viewed or not'''
    
    print("START -----> matrixBuilder " + str(datetime.now()))
    rowSesID = []
    colItemID = []
    val =[]
    
    for ses,items in itemsdict.items():
        for item in items:
            if (item != 643078800 and item != 214685805 and item != 214821302):
                rowSesID.append(ses)
                colItemID.append(item)
                val.append(itemsdict[ses][item])

    print("END -----> matrixBuilder " + str(datetime.now()))
    
    return mat 
   
def main(testRunFlag):
    '''main functions. executes call to other functions. divided by first "if" which determines False for first run and True for the second run'''
    print("START ... " + str(datetime.now()))
    
    global sessionsFeatures;global itemsdict
    
    if(testRunFlag == True):
        
        # this is the prediction and recommender phase
        if(os.path.exists(sessionsFeaturesFileTest)):
            # Load the sessionFeatures dictionary from dump
            sessionsFeatures = read_object_from_file(sessionsFeaturesFileTest)
             
            #itemsdict = read_object_from_file(itemsdictFileTest)
            timeSpentItemDictList= read_object_from_file(timespentdictFileTest)
        else:
            # Load the test data 
            testData = loadData(testFileName, delimiter = ',', dtype = {'names': ('sesId', 'TS', 'itemId', 'category'),'formats': ('i4', 'S24', 'i4', 'S4')})
            
            # Populate sessionFeatures and itemsdict
            extractFeatures(testData)
            # Scale the features:
            keys,vals = zip(*sessionsFeatures.items())
            vals = dataPreparation(np.asarray(vals))
            sessionsFeatures = dict(zip(keys,vals))
            # Write the sessionFeatures matrix to file 
            write_object_to_file(sessionsFeatures,sessionsFeaturesFileTest)
            
            # Write the itemsdict to file for further analysis:
            write_object_to_file(itemsdict,itemsdictFileTest)
            write_object_to_file(timeSpentItemDictList,timespentdictFileTest)
          
        # Predict which sessions will buy
        test = np.asarray(sessionsFeatures.values())[:,0:-1]
        
        buyers = predict(test)
        # Return the real session ids of sessions that we predicted that will buy
        buyersIds = list(np.asarray(sessionsFeatures.keys())[buyers])
        
        #recommendSessions(buyersIds)
        #naiveReccomender(buyersIds,k=5)
        TimeSpentRecommender(buyersIds)
        
    else: # This is the training phase
        
        
        if(os.path.exists(sessionsFeaturesFileTrain)):
            # Load the sessionFeatures dictionary from dump
            sessionsFeatures = read_object_from_file(sessionsFeaturesFileTrain)
                
        else:
            
            # Load the source data and extract the features:
            clickData = loadData(clickFileName, delimiter = ',', dtype = {'names': ('sesId', 'TS', 'itemId', 'category'),'formats': ('i4', 'S24', 'i4', 'S4')})
            buyData = loadData(buyFileName, delimiter = ',', dtype = {'names': ('sesId', 'TS', 'itemId', 'price', 'quantity'),'formats': ('i4', 'S24', 'i4', 'f4', 'f4')})
            
            # Populate sessionFeatures and itemsdict
            extractFeatures(clickData)
            
            # for unsupervised learning 
            createLabelForClassification(clickData, buyData)
            
            # Scale the features:
            keys,vals = zip(*sessionsFeatures.items())
            vals = dataPreparation(np.asarray(vals))
            sessionsFeatures = dict(zip(keys,vals))
            # Write the sessionFeatures matrix to file 
            write_object_to_file(sessionsFeatures,sessionsFeaturesFileTrain)
            
          
        # Training the classification model: if session "is bought" or not 
        classifySessions(np.asarray(sessionsFeatures.values()))
        
     
    print("END ! " + str(datetime.now()))
    
    return

main(testRunFlag=False)
