import sys
import csv
import string
import random

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(threshold='nan')

#############################GLOBAL VARIABLES##################################
#The following three global variables check
#which setting the system should run on
#Usual is the setting in which there is no evaluation
#that are done for Q3 AND Q4.
#ExploreTrainingSetSize is 1 when we do Q3
#ExploreFeatureSpace is 1 when we want to do Q4
#EIther one of the three booleans are 1 at a time.
UsualSetting = 1
ExploreModelChoice = 0
ExploreFeatureSpace = 0

#Following are settings under which we want to run the
#system. If ExploreTrainingSetSize is 1 then percentTraining
#is used to explore different percentages of training size
#This value is set to 50% in case we are running ExploreFeatureSpace
#numWordsToDiscardTop is always 100.
#numWordsToConstructFeat is to explore the feature set space.
#It is set to a constant 500 when we want to run the system
#uunder ExploreTrainingSetSize setting.
percentTraining = [1, 3, 5, 8, 10, 15]
# percentTraining = [15]
numWordsToDiscardTop = 100
numWordsToConstructFeat = 4000
numFolds=10
foldSize=200

#parameters of LR
LRLAMBDA = 0.01
LRETA = 0.01

#parameters of SVM
SVMLAMBDA = 0.01
SVMETA = 0.5

NUMITERS = 100
TOLERANCE = 1e-6

#############################COMMON FUNCTIONS FOR ALL MODELS##################################

def readCSVFile(trainingFileName):
    csvfile = open(trainingFileName, 'rb')
    csvContent = [row for row in csv.reader\
                    (csvfile.read().splitlines(), delimiter='\t')]
    return csvContent

def getUniqueWords(csvContent):
    uniqueWordsAndFreqs = {}

    for i in range(0,len(csvContent)):
        reviewText = csvContent[i][2]
        reviewText =  list(set(reviewText)) #get unique words in the reviewText
        # row[2] = reviewText
        for word in reviewText: #go over each word in the review and find its entry in unique words hashtable.
                            #if there is an entry just icrement frequency by 1. otherwise, create an entry
                            #and set value to 1.
            if word not in uniqueWordsAndFreqs:
                uniqueWordsAndFreqs[word] = 1
            else:
                uniqueWordsAndFreqs[word] = uniqueWordsAndFreqs.get(word) + 1

    uniqueWordsFreqsTuple = sorted(uniqueWordsAndFreqs.items(), key=lambda x:x[1], reverse=True) #sort words
                                                                                            # based on frequncies
    return uniqueWordsFreqsTuple

def printTOP10Words(words):
    countWords = 0
    for word in words[:11]:
        print 'WORD' + str(countWords) + ' ' + word
        countWords += 1

def constructFeatures(content,listWords):

    features = []
    for i in range(len(listWords)):
        features.append([0]*len(content)) #initialize list of size number of features times
                                            # number of reviews to all 0s

    for i in range(0,len(listWords)): # go to each word in the list
        word = listWords[i]
        for j in range(0,len(content)): # go to each reviewText in the set of reviews
            reviewContent = content[j][2]
            if word in reviewContent: #see if the word appears in the review
                features[i][j] = 1 #put a one there if it does

    return np.array(features)

def constructFeaturesThreeVals(content,listWords):

    features = []
    for i in range(len(listWords)):
        features.append([0]*len(content)) #initialize list of size number of features times
                                            # number of reviews to all 0s

    for i in range(0,len(listWords)): # go to each word in the list
        word = listWords[i]
        for j in range(0,len(content)): # go to each reviewText in the set of reviews
            reviewContent = content[j][2]
            numTimesWordOccur = reviewContent.count(word)
            if numTimesWordOccur == 1:
                features[i][j] = 1 #put a one there if it does
            elif numTimesWordOccur > 1:
                features[i][j] = 2

    return np.array(features)

def getClassLabels(content):
    labels = []
    for row in content:
        labels.append(int(row[1]))
    return np.array(labels)

def findZeroOneLoss(a,b):
    sum = 0.0
    for i in range(0,len(a)):
        if a[i] != b[i]:
            sum += 1.0
    sum = sum / float(len(a))
    return sum

def average(listNums):
    return float(sum(listNums)) / max(len(listNums), 1)

def getRandomIDList(t):
    listAll = range(0,t)
    random.shuffle(listAll)
    randomIDList = []
    for i in range(0,numFolds):
        randomIDList.append(listAll[i*foldSize:i*foldSize+foldSize])
    return randomIDList

def getNorm2(v1,v2):
    return np.linalg.norm(v1-v2)

#############################FUNCTIONS FOR NBC##################################

def getClassLabelsNBC(content):
    labels = []
    for row in content:
        labels.append(int(row[1]))
    return labels

def constructFeaturesNBC(content,listWords):

    features = []
    for i in range(len(listWords)):
        features.append([0]*len(content)) #initialize list of size number of features times number of reviews to all 0s

    countWords = 0
    for word in listWords: # go to each word in the list
        countReview = 0
        for row in content: # go to each reviewText in the set of reviews
            reviewContent = row[2]
            if word in reviewContent: #see if the word appears in the review
                features[countWords][countReview] = 1 #put a one there if it does
            countReview += 1
        countWords += 1
    return features

def constructFeaturesNBCThreeVals(content,listWords):

    features = []
    for i in range(len(listWords)):
        features.append([0]*len(content)) #initialize list of size number of features times number of reviews to all 0s

    countWords = 0
    for word in listWords: # go to each word in the list
        countReview = 0
        for row in content: # go to each reviewText in the set of reviews
            reviewContent = row[2]
            numTimesWordOccur = reviewContent.count(word)
            if numTimesWordOccur == 1: #see if the word appears in the review
                features[countWords][countReview] = 1 #put a one there if it does
            elif numTimesWordOccur > 1:
                features[countWords][countReview] = 2
            countReview += 1
        countWords += 1
    return features

def getPriorDist(content):
    numOnes = 0.0
    for row in content: #go over each review
        if int(row[1]) > 0: #find if the label is one (i.e. > 0)
            numOnes += 1.0 #increment count of number of ones
    return [1.0-numOnes/len(content), numOnes/len(content)] #first element is prior that label is 0,
                                                            # second for label = 1

def getCPDS(features,labels):
    cpds0 = []
    cpds1 = []
    for i in range(len(features)):
        cpds0.append([0, 0])
        cpds1.append([0, 0])

    labels0idx = [i for i, e in enumerate(labels) if e == 0]
    labels1idx = [i for i, e in enumerate(labels) if e != 0]

    numLabel0 = len(labels0idx)
    numLabel1 = len(labels1idx)

    countFeature = 0
    for f in features:
        numTimesf0l0 = 0.0
        numTimesf1l0 = 0.0
        for id0 in labels0idx:
            if f[id0] == 0:
                numTimesf0l0 += 1.0
            else:
                numTimesf1l0 += 1.0

        numTimesf0l1 = 0.0
        numTimesf1l1 = 0.0
        for id1 in labels1idx:
            if f[id1] == 0:
                numTimesf0l1 += 1.0
            else:
                numTimesf1l1 += 1.0

        cpds0[countFeature][0] = (numTimesf0l0 + 1) / float(numLabel0 + 2)
        cpds0[countFeature][1] = (numTimesf1l0 + 1) / float(numLabel0 + 2)
        cpds1[countFeature][0] = (numTimesf0l1 + 1) / float(numLabel1 + 2)
        cpds1[countFeature][1] = (numTimesf1l1 + 1) / float(numLabel1 + 2)

        countFeature += 1
    cpds = [cpds0, cpds1]
    return cpds

def getCPDSThreeVals(features,labels):
    cpds0 = []
    cpds1 = []
    for i in range(len(features)):
        cpds0.append([0, 0, 0])
        cpds1.append([0, 0, 0])

    labels0idx = [i for i, e in enumerate(labels) if e == 0]
    labels1idx = [i for i, e in enumerate(labels) if e != 0]

    numLabel0 = len(labels0idx)
    numLabel1 = len(labels1idx)

    countFeature = 0
    for f in features:
        numTimesf0l0 = 0.0
        numTimesf1l0 = 0.0
        numTimesf2l0 = 0.0
        for id0 in labels0idx:
            if f[id0] == 0:
                numTimesf0l0 += 1.0
            elif f[id0] == 1:
                numTimesf1l0 += 1.0
            elif f[id0] == 2:
                numTimesf2l0 += 1.0

        numTimesf0l1 = 0.0
        numTimesf1l1 = 0.0
        numTimesf2l1 = 0.0
        for id1 in labels1idx:
            if f[id1] == 0:
                numTimesf0l1 += 1.0
            elif f[id1] == 1:
                numTimesf1l1 += 1.0
            elif f[id1] == 2:
                numTimesf2l1 += 1.0

        cpds0[countFeature][0] = (numTimesf0l0 + 1) / float(numLabel0 + 3)
        cpds0[countFeature][1] = (numTimesf1l0 + 1) / float(numLabel0 + 3)
        cpds0[countFeature][2] = (numTimesf2l0 + 1) / float(numLabel0 + 3)
        cpds1[countFeature][0] = (numTimesf0l1 + 1) / float(numLabel1 + 3)
        cpds1[countFeature][1] = (numTimesf1l1 + 1) / float(numLabel1 + 3)
        cpds1[countFeature][2] = (numTimesf2l1 + 1) / float(numLabel1 + 3)

        countFeature += 1
    cpds = [cpds0, cpds1]
    return cpds

def predictUsingNBC(features,cpds,prior):
    numFeatures = len(features)
    numReviews2Test = len(features[0])

    labels = [0 for x in range(0,numReviews2Test)]
    condPr1 = [1.0 for y in range(0,numReviews2Test)]
    condPr0 = [1.0 for z in range(0,numReviews2Test)]

    for i in range(0,numFeatures):
        for j in range(0,numReviews2Test):
            if features[i][j] == 0:
                condPr1[j] *= cpds[1][i][0]
                condPr0[j] *= cpds[0][i][0]
            if features[i][j] == 1:
                condPr1[j] *= cpds[1][i][1]
                condPr0[j] *= cpds[0][i][1]

    for i in range(0,len(condPr1)):
        condPr1[i] *= prior[1]
        condPr0[i] *= prior[0]
        if condPr1[i] >= condPr0[i]:
            labels[i] = 1
    return labels

def predictUsingNBCThreeVals(features,cpds,prior):
    numFeatures = len(features)
    numReviews2Test = len(features[0])

    labels = [0 for x in range(0,numReviews2Test)]
    condPr1 = [1.0 for y in range(0,numReviews2Test)]
    condPr0 = [1.0 for z in range(0,numReviews2Test)]

    for i in range(0,numFeatures):
        for j in range(0,numReviews2Test):
            if features[i][j] == 0:
                condPr1[j] *= cpds[1][i][0]
                condPr0[j] *= cpds[0][i][0]
            if features[i][j] == 1:
                condPr1[j] *= cpds[1][i][1]
                condPr0[j] *= cpds[0][i][1]
            if features[i][j] == 2:
                condPr1[j] *= cpds[1][i][2]
                condPr0[j] *= cpds[0][i][2]

    for i in range(0,len(condPr1)):
        condPr1[i] *= prior[1]
        condPr0[i] *= prior[0]
        if condPr1[i] >= condPr0[i]:
            labels[i] = 1
    return labels

#############################FUNCTIONS FOR LR##################################

def getFeatValuesForEx(features, countEx):
    return np.array([float(l[countEx]) for l in features])

def logisticFunc(w,x):
    return 1.0/(1.0 + np.exp(-(np.dot(w,x))))

def getGradValsLR(p,t,f,w):
    deljList = np.zeros(shape=(len(w)), dtype=float)
    for j in range(0,len(w)):
        delj = 0.0
        for i in range(0,len(f[0])):
            delj += (t[i] - p[i]) * f[j][i]
        delj -= LRLAMBDA * w[j]
        deljList[j] = delj
    return deljList

def getLRWeights(features,trueLabels):

    w = np.zeros(shape=(len(features)),dtype=float)
    wprev = np.ones(shape=(len(features)),dtype=float)

    for i in range(0,NUMITERS):
        predLabels = np.zeros(shape=(len(features[0])),dtype=float)

        for countEx in range(0,len(features[0])):
            x = getFeatValuesForEx(features, countEx)
            predLabels[countEx] = logisticFunc(w,x)
            # print trueLabels[countEx], predLabels[countEx]

        gradVector = getGradValsLR(predLabels,trueLabels,features,w)
        gradVector = LRETA * gradVector
        w = np.add(w,gradVector)

        normVal = getNorm2(w, wprev)
        if normVal <= TOLERANCE:
            break
        wprev = w
    # print w
    # for countEx in range(0,len(features[0])):
    #     print trueLabels[countEx], predLabels[countEx]
    return w

def predictUsingLR(features,w):
    predLabel =[]
    for countEx in range(0,len(features[0])):
        x = getFeatValuesForEx(features, countEx)
        logVal = logisticFunc(w,x)
        if logVal >= 0.5:
            predLabel.append(1)
        else:
            predLabel.append(0)
    return predLabel

#######################################SVM FUNCS#################################


def getGradValsSVM(p,t,f,w):
    deljList = np.zeros(shape=(len(w)), dtype=float)
    for j in range(0,len(w)):
        delj = 0.0
        for i in range(0,len(f[0])):
            delji = t[i] * f[j][i] if (p[i]*t[i] < 1) else 0.0
            delj += SVMLAMBDA * w[j] - delji
        delj = delj / len(p)
        deljList[j] = delj
    return deljList

def getSVMWeights(features,trueLabels):

    w = np.zeros(shape=(len(features)),dtype=float)
    wprev = np.ones(shape=(len(features)),dtype=float)

    for i in range(0,NUMITERS):
        predLabels = np.zeros(shape=(len(features[0])),dtype=float)

        for countEx in range(0,len(features[0])):
            x = getFeatValuesForEx(features, countEx)
            predLabels[countEx] = np.dot(w,x)

        gradVector = getGradValsSVM(predLabels,trueLabels,features,w)
        gradVector = SVMETA * gradVector
        w = np.subtract(w,gradVector)

        normVal = getNorm2(w, wprev)
        # print i, normVal
        if normVal <= TOLERANCE:
            break
        wprev = w

    # for countEx in range(0,len(features[0])):
    #     print trueLabels[countEx], predLabels[countEx]
    return w

def predictUsingSVM(features,w):
    predLabel =[]
    for countEx in range(0,len(features[0])):
        x = getFeatValuesForEx(features, countEx)
        predVal = np.dot(w,x)
        if predVal > 0.0:
            predLabel.append(1.0)
        else:
            predLabel.append(-1.0)
    return predLabel

#######################################MAIN FUNC#################################

def main(arg):

    if ExploreModelChoice:

        trainingFileName = arg[1]
        csvContent = readCSVFile(trainingFileName)
        randIDLists = getRandomIDList(len(csvContent))

        for countRevs in range(0,len(csvContent)):
            csvContent[countRevs][2] = csvContent[countRevs][2].lower().translate(None, string.punctuation).split()

        avgLLRs = []
        avgLSVMs = []
        avgLNBCs = []
        seLLRs = []
        seLSVMs = []
        seLNBCs = []
        TSS = []
        for counterSizes in range(0,len(percentTraining)):
            print "======================================="
            LLR = []
            LSVM = []
            LNBC = []
            for fold in range(0,numFolds):

                testSetIDs = randIDLists[fold]
                restLists = randIDLists[0:fold]+randIDLists[fold+1:]
                restIDs = [item for sublist in restLists for item in sublist]

                numTrainingEx = int((percentTraining[counterSizes] / 100.0) * len(csvContent))
                randIdsTr = random.sample(restIDs, numTrainingEx)

                if fold == 0:
                    TSS.append(numTrainingEx)

                trainingContent = []
                testingContent = []

                for i in range(0,len(randIdsTr)):
                    temp = csvContent[randIdsTr[i]]
                    trainingContent.append(temp)
                for i in range(0,len(testSetIDs)):
                    temp = csvContent[testSetIDs[i]]
                    testingContent.append(temp)

                uniqueWordsFreqsTuple = getUniqueWords(trainingContent)
                # dummy = getUniqueWords(testingContent) #this dummy is never used.
                uniqueWordsFreqsTuple = uniqueWordsFreqsTuple\
                    [numWordsToDiscardTop:numWordsToDiscardTop+numWordsToConstructFeat]
                listWords = [i for (i,j) in uniqueWordsFreqsTuple]

                featuresTraining = constructFeatures(trainingContent, listWords)
                featuresTraining = featuresTraining.astype(float)

                featuresTesting = constructFeatures(testingContent, listWords)
                featuresTesting = featuresTesting.astype(float)

                classLabelsTr = getClassLabels(trainingContent)
                classLabelsTr = classLabelsTr.astype(float)

                classLabelsTs = getClassLabels(testingContent)
                classLabelsTs = classLabelsTs.astype(float)

                # if int(modelChoice) == 1: #LR

                featuresTraining = np.vstack([np.ones(shape=(1,len(featuresTraining[0]))),featuresTraining])
                featuresTesting = np.vstack([np.ones(shape=(1,len(featuresTesting[0]))), featuresTesting])

                LRWeights = getLRWeights(featuresTraining, classLabelsTr)
                predLabelsLR = predictUsingLR(featuresTesting,LRWeights)

                zeroOneLossLR = findZeroOneLoss(predLabelsLR,classLabelsTs)
                print 'ZERO-ONE-LOSS-LR ' + str(zeroOneLossLR)
                LLR.append(zeroOneLossLR)

                # elif int(modelChoice) == 2: #SVM
                classLabelsTr[classLabelsTr==0.0] = -1.0
                SVMWeights = getSVMWeights(featuresTraining, classLabelsTr)
                predLabelsSVM = predictUsingSVM(featuresTesting,SVMWeights)
                predLabelsSVM = np.array(predLabelsSVM)
                classLabelsTs[classLabelsTs==0.0] = -1.0

                zeroOneLossSVM = findZeroOneLoss(predLabelsSVM,classLabelsTs)
                print 'ZERO-ONE-LOSS-SVM ' + str(zeroOneLossSVM)
                LSVM.append(zeroOneLossSVM)

                featuresTrainingNBC = constructFeaturesNBC(trainingContent, listWords)
                featuresTestingNBC = constructFeaturesNBC(testingContent, listWords)

                classLabelsTrNBC = getClassLabelsNBC(trainingContent)
                classLabelsTsNBC = getClassLabelsNBC(testingContent)

                prior = getPriorDist(trainingContent)
                cpds = getCPDS(featuresTrainingNBC, classLabelsTrNBC)

                predLabelsNBC = predictUsingNBC(featuresTestingNBC,cpds,prior)

                zeroOneLossNBC = findZeroOneLoss(predLabelsNBC,classLabelsTsNBC)
                print 'ZERO-ONE-LOSS-NBC ' + str(zeroOneLossNBC)
                LNBC.append(zeroOneLossNBC)

            avgLLR = average(LLR)
            avgLSVM = average(LSVM)
            avgLNBC = average(LNBC)

            avgLLRs.append(avgLLR)
            avgLSVMs.append(avgLSVM)
            avgLNBCs.append(avgLNBC)

            stdLLR = np.std(LLR)
            stdLSVM = np.std(LSVM)
            stdLNBC = np.std(LNBC)

            seLLR = stdLLR / float(numFolds)
            seLSVM = stdLSVM / float(numFolds)
            seLNBC = stdLNBC / float(numFolds)

            seLLRs.append(seLLR)
            seLSVMs.append(seLSVM)
            seLNBCs.append(seLNBC)

        print "=======LR======"
        print avgLLRs
        print seLLRs

        print "=======SVM======"
        print avgLSVMs
        print seLSVMs

        print "=======NBC======"
        print avgLNBCs
        print seLNBCs

        plt.figure()
        plt.errorbar(TSS, avgLLRs, seLLRs, label='LR')
        plt.errorbar(TSS, avgLSVMs, seLSVMs, label='SVM')
        plt.errorbar(TSS, avgLNBCs, seLNBCs, label='NBC')
        plt.ylabel('Zero-one Loss')
        plt.xlabel('Training set sizes')
        plt.title('Zero One loss for different training set size ')
        plt.legend()
        axes = plt.gca()
        axes.set_ylim([0,0.6])
        plt.savefig('AllModelsBinaryFeats.png')
        plt.show()
        plt.close()



    if ExploreFeatureSpace:

        trainingFileName = arg[1]
        csvContent = readCSVFile(trainingFileName)
        randIDLists = getRandomIDList(len(csvContent))

        for countRevs in range(0,len(csvContent)):
            csvContent[countRevs][2] = csvContent[countRevs][2].lower().translate(None, string.punctuation).split()

        avgLLRs = []
        avgLSVMs = []
        avgLNBCs = []
        avgLLR2vals = []
        seLLRs = []
        seLSVMs = []
        seLNBCs = []
        seLLR2vals = []
        TSS = []
        for counterSizes in range(0,len(percentTraining)):
            print "======================================="
            LLR = []
            LSVM = []
            LNBC = []
            LLR2val = []
            for fold in range(0,numFolds):

                testSetIDs = randIDLists[fold]
                restLists = randIDLists[0:fold]+randIDLists[fold+1:]
                restIDs = [item for sublist in restLists for item in sublist]

                numTrainingEx = int((percentTraining[counterSizes] / 100.0) * len(csvContent))
                randIdsTr = random.sample(restIDs, numTrainingEx)

                if fold == 0:
                    TSS.append(numTrainingEx)

                trainingContent = []
                testingContent = []

                for i in range(0,len(randIdsTr)):
                    temp = csvContent[randIdsTr[i]]
                    trainingContent.append(temp)
                for i in range(0,len(testSetIDs)):
                    temp = csvContent[testSetIDs[i]]
                    testingContent.append(temp)

                uniqueWordsFreqsTuple = getUniqueWords(trainingContent)
                # dummy = getUniqueWords(testingContent) #this dummy is never used.
                uniqueWordsFreqsTuple = uniqueWordsFreqsTuple\
                    [numWordsToDiscardTop:numWordsToDiscardTop+numWordsToConstructFeat]
                listWords = [i for (i,j) in uniqueWordsFreqsTuple]

                featuresTraining = constructFeaturesThreeVals(trainingContent, listWords)
                featuresTraining = featuresTraining.astype(float)

                featuresTesting = constructFeaturesThreeVals(testingContent, listWords)
                featuresTesting = featuresTesting.astype(float)

                classLabelsTr = getClassLabels(trainingContent)
                classLabelsTr = classLabelsTr.astype(float)

                classLabelsTs = getClassLabels(testingContent)
                classLabelsTs = classLabelsTs.astype(float)

                # if int(modelChoice) == 1: #LR

                featuresTraining = np.vstack([np.ones(shape=(1,len(featuresTraining[0]))),featuresTraining])
                featuresTesting = np.vstack([np.ones(shape=(1,len(featuresTesting[0]))), featuresTesting])

                LRWeights = getLRWeights(featuresTraining, classLabelsTr)
                predLabelsLR = predictUsingLR(featuresTesting,LRWeights)

                zeroOneLossLR = findZeroOneLoss(predLabelsLR,classLabelsTs)
                print 'ZERO-ONE-LOSS-LR ' + str(zeroOneLossLR)
                LLR.append(zeroOneLossLR)

                #2 value feature for lr
                featuresTraining2val = constructFeatures(trainingContent, listWords)
                featuresTraining2val = featuresTraining2val.astype(float)

                featuresTesting2val = constructFeatures(testingContent, listWords)
                featuresTesting2val = featuresTesting2val.astype(float)

                # if int(modelChoice) == 1: #LR
                featuresTraining2val = np.vstack([np.ones(shape=(1,len(featuresTraining2val[0]))),featuresTraining2val])
                featuresTesting2val = np.vstack([np.ones(shape=(1,len(featuresTesting2val[0]))), featuresTesting2val])

                LRWeights2val = getLRWeights(featuresTraining2val, classLabelsTr)
                predLabelsLR2val = predictUsingLR(featuresTesting2val,LRWeights2val)

                zeroOneLossLR2val = findZeroOneLoss(predLabelsLR2val,classLabelsTs)
                print 'ZERO-ONE-LOSS-LR-2val ' + str(zeroOneLossLR2val)
                LLR2val.append(zeroOneLossLR2val)

                # elif int(modelChoice) == 2: #SVM
                classLabelsTr[classLabelsTr==0.0] = -1.0
                SVMWeights = getSVMWeights(featuresTraining, classLabelsTr)
                predLabelsSVM = predictUsingSVM(featuresTesting,SVMWeights)
                predLabelsSVM = np.array(predLabelsSVM)
                classLabelsTs[classLabelsTs==0.0] = -1.0

                zeroOneLossSVM = findZeroOneLoss(predLabelsSVM,classLabelsTs)
                print 'ZERO-ONE-LOSS-SVM ' + str(zeroOneLossSVM)
                LSVM.append(zeroOneLossSVM)

                featuresTrainingNBC = constructFeaturesNBCThreeVals(trainingContent, listWords)
                featuresTestingNBC = constructFeaturesNBCThreeVals(testingContent, listWords)

                classLabelsTrNBC = getClassLabelsNBC(trainingContent)
                classLabelsTsNBC = getClassLabelsNBC(testingContent)

                prior = getPriorDist(trainingContent)
                cpds = getCPDSThreeVals(featuresTrainingNBC, classLabelsTrNBC)

                predLabelsNBC = predictUsingNBCThreeVals(featuresTestingNBC,cpds,prior)

                zeroOneLossNBC = findZeroOneLoss(predLabelsNBC,classLabelsTsNBC)
                print 'ZERO-ONE-LOSS-NBC ' + str(zeroOneLossNBC)
                LNBC.append(zeroOneLossNBC)


            avgLLR = average(LLR)
            avgLSVM = average(LSVM)
            avgLNBC = average(LNBC)
            avgLLR2val = average(LLR2val)

            avgLLRs.append(avgLLR)
            avgLSVMs.append(avgLSVM)
            avgLNBCs.append(avgLNBC)
            avgLLR2vals.append(avgLLR2val)

            stdLLR = np.std(LLR)
            stdLSVM = np.std(LSVM)
            stdLNBC = np.std(LNBC)
            stdLLR2val = np.std(LLR2val)

            seLLR = stdLLR / float(numFolds)
            seLSVM = stdLSVM / float(numFolds)
            seLNBC = stdLNBC / float(numFolds)
            seLLR2val = stdLLR2val / float(numFolds)

            seLLRs.append(seLLR)
            seLSVMs.append(seLSVM)
            seLNBCs.append(seLNBC)
            seLLR2vals.append(seLLR2val)

        print "=======LR======"
        print avgLLRs
        print seLLRs

        print "=======SVM======"
        print avgLSVMs
        print seLSVMs

        print "=======NBC======"
        print avgLNBCs
        print seLNBCs

        print "=======LR2val======"
        print avgLLR2vals
        print seLLR2vals

        plt.figure()
        plt.errorbar(TSS, avgLLRs, seLLRs, label='LR')
        plt.errorbar(TSS, avgLSVMs, seLSVMs,label='SVM')
        plt.errorbar(TSS, avgLNBCs, seLNBCs, label='NBC')
        plt.ylabel('Zero-one Loss')
        plt.xlabel('Training set sizes')
        plt.title('Zero One loss for different training set size ')
        plt.legend()
        axes = plt.gca()
        axes.set_ylim([0,0.6])
        plt.savefig('AllModelsBinaryFeats.png')
        plt.show()
        plt.close()

    if UsualSetting:

        #from trainingFileName and testingfilename given as input read in the csv file and
        #store in variable.
        trainingFileName = arg[1]
        trainingContent = readCSVFile(trainingFileName)
        testingFileName = arg[2]
        testingContent = readCSVFile(testingFileName)
        modelChoice = arg[3] # is 1 for LR, 2 for  SVM, and 3 for NBC

        for countRevs in range(0,len(trainingContent)):
            trainingContent[countRevs][2] = trainingContent[countRevs][2].lower().translate(None, string.punctuation).split()
        for countRevs in range(0,len(testingContent)):
            testingContent[countRevs][2] = testingContent[countRevs][2].lower().translate(None, string.punctuation).split()

        #get the unique words from the training content.
        #we get a sorted list of word in a decreasing order wrt
        #the collection frequency. Top numWordsToDiscardTop words
        #are discarded as they are really high frequency and are
        #proabably stopwords. Then we consider only numWordsToConstructFeat
        #number of words.
        uniqueWordsFreqsTuple = getUniqueWords(trainingContent)
        # dummy = getUniqueWords(testingContent) #this dummy is never used.
                                               # Just to split words of testing content.
        uniqueWordsFreqsTuple = uniqueWordsFreqsTuple\
            [numWordsToDiscardTop:numWordsToDiscardTop+numWordsToConstructFeat]
        listWords = [i for (i,j) in uniqueWordsFreqsTuple]

        #Afterwards we construct features from the training and testing
        #content. Specifically, these features set is a matrix of
        #size number of words considered times number of reviews in content.

        if int(modelChoice) == 1: #LR
            featuresTraining = constructFeatures(trainingContent, listWords)
            featuresTraining = featuresTraining.astype(float)

            featuresTesting = constructFeatures(testingContent, listWords)
            featuresTesting = featuresTesting.astype(float)

            classLabelsTr = getClassLabels(trainingContent)
            classLabelsTr = classLabelsTr.astype(float)

            classLabelsTs = getClassLabels(testingContent)
            classLabelsTs = classLabelsTs.astype(float)

            featuresTraining = np.vstack([np.ones(shape=(1,len(featuresTraining[0]))),featuresTraining])
            featuresTesting = np.vstack([np.ones(shape=(1,len(featuresTesting[0]))), featuresTesting])

            LRWeights = getLRWeights(featuresTraining, classLabelsTr)
            predLabelsLR = predictUsingLR(featuresTesting,LRWeights)

            zeroOneLossLR = findZeroOneLoss(predLabelsLR,classLabelsTs)
            print 'ZERO-ONE-LOSS-LR ' + str(zeroOneLossLR)

        if int(modelChoice) == 2: #SVM
            featuresTraining = constructFeatures(trainingContent, listWords)
            featuresTraining = featuresTraining.astype(float)

            featuresTesting = constructFeatures(testingContent, listWords)
            featuresTesting = featuresTesting.astype(float)

            classLabelsTr = getClassLabels(trainingContent)
            classLabelsTr = classLabelsTr.astype(float)

            classLabelsTs = getClassLabels(testingContent)
            classLabelsTs = classLabelsTs.astype(float)

            featuresTraining = np.vstack([np.ones(shape=(1,len(featuresTraining[0]))),featuresTraining])
            featuresTesting = np.vstack([np.ones(shape=(1,len(featuresTesting[0]))), featuresTesting])

            classLabelsTr[classLabelsTr==0.0] = -1.0
            SVMWeights = getSVMWeights(featuresTraining, classLabelsTr)
            predLabelsSVM = predictUsingSVM(featuresTesting,SVMWeights)
            predLabelsSVM = np.array(predLabelsSVM)
            classLabelsTs[classLabelsTs==0.0] = -1.0

            zeroOneLossSVM = findZeroOneLoss(predLabelsSVM,classLabelsTs)
            print 'ZERO-ONE-LOSS-SVM ' + str(zeroOneLossSVM)

if __name__ == "__main__":
    main(sys.argv[0:])
