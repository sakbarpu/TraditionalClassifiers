import sys
import csv
import string
import random
import matplotlib.pyplot as plt
import numpy as np


#The following three global variables check
#which setting the system should run on
#Usual is the setting in which there is no evaluation
#that are done for Q3 AND Q4.
#ExploreTrainingSetSize is 1 when we do Q3
#ExploreFeatureSpace is 1 when we want to do Q4
#EIther one of the three booleans are 1 at a time.
UsualSetting = 1
ExploreTrainingSetSize = 0
ExploreFeatureSpace = 0

#Following are settings under which we want to run the
#system. If ExploreTrainingSetSize is 1 then percentTraining
#is used to explore different percentages of training size
#This value is set to 50% in case we are running ExploreFeatureSpace
#numWordsToDiscardTop is always 100.
#numWordsToConstructFeat is to explore the feature set space.
#It is set to a constant 500 when we want to run the system
#uunder ExploreTrainingSetSize setting.
percentTraining = [1, 5, 10, 20, 50, 90]
numWordsToDiscardTop = 100
numWordsToConstructFeat = [10, 50, 250, 500, 1000, 4000]

def readCSVFile(trainingFileName):
    csvfile = open(trainingFileName, 'rb')
    csvContent = [row for row in csv.reader\
                    (csvfile.read().splitlines(), delimiter='\t')]
    return csvContent

def getUniqueWords(csvContent):
    uniqueWordsAndFreqs = {}

    for row in csvContent:
        reviewText = row[2].lower().translate(None, string.punctuation).split() #convert to lowercase, remove punctuations, and split
        reviewText =  list(set(reviewText)) #get unique words in the reviewText
        row[2] = reviewText
        for word in reviewText: #go over each word in the review and find its entry in unique words hashtable.
                            #if there is an entry just icrement frequency by 1. otherwise, create an entry
                            #and set value to 1.
            if word not in uniqueWordsAndFreqs:
                uniqueWordsAndFreqs[word] = 1
            else:
                uniqueWordsAndFreqs[word] = uniqueWordsAndFreqs.get(word) + 1

    uniqueWordsFreqsTuple = sorted(uniqueWordsAndFreqs.items(), key=lambda x:x[1], reverse=True) #sort words based on frequncies
    return uniqueWordsFreqsTuple

def printTOP10Words(words):
    countWords = 0
    for word in words[:11]:
        print 'WORD' + str(countWords) + ' ' + word
        countWords += 1

def constructFeatures(content,listWords):

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

def getPriorDist(content):
    numOnes = 0.0
    for row in content: #go over each review
        if int(row[1]) > 0: #find if the label is one (i.e. > 0)
            numOnes += 1.0 #increment count of number of ones
    return [1.0-numOnes/len(content), numOnes/len(content)] #first element is prior that label is 0, second for label = 1

def getClassLabels(content):
    labels = []
    for row in content:
        labels.append(int(row[1]))
    return labels

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

def testSystem(features,cpds,prior):
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

def findZeroOneLoss(a,b):
    return float(sum([abs(i - j) for i, j in zip(a, b)])) / float(len(a))

def average(listNums):
    return float(sum(listNums)) / max(len(listNums), 1)

def main(arg):

    if ExploreTrainingSetSize:
        avgZeroOneLosses = []
        avgBaseLineLosses = []
        trainingSetSize = []
        for counterSizes in range(0,len(percentTraining)):

            zeroOneLossList = []
            baslineLossList = []
            numTrials = []

            for trials in range(0,10):

                numTrials.append(trials)
                #from trainingFileName given as input read in the csv file and
                #store in csvContent variable.
                trainingFileName = arg[1]
                csvContent = readCSVFile(trainingFileName)

                #from all the csv content for the training and testing content
                #based on the random sampling mentioned in the handout.
                #this uses the percentTraining which is a global parameter
                #defining the percent of csvContent to be used for training.
                #random sampling is done for figuring out which reviews form
                #training set. The rest goes to testing set.
                numTrainingEx = int((percentTraining[counterSizes] / 100.0) * len(csvContent))
                listAllIdsContent = range(0,len(csvContent))
                randIdsTr = random.sample(listAllIdsContent, numTrainingEx)
                randIdsTs = list(set(listAllIdsContent)- set(randIdsTr))
                trainingContent = [csvContent[i] for i in randIdsTr]
                testingContent = [csvContent[i] for i in randIdsTs]
                if trials == 0:
                    trainingSetSize.append(numTrainingEx)

                #get the unique words from the training content.
                #we get a sorted list of word in a decreasing order wrt
                #the collection frequency. Top numWordsToDiscardTop words
                #are discarded as they are really high frequency and are
                #proabably stopwords. Then we consider only numWordsToConstructFeat
                #number of words. Afterwards we print the top 10 words.
                #If there is a tie in the top 10 words we print both.
                uniqueWordsFreqsTuple = getUniqueWords(trainingContent)
                dummy = getUniqueWords(testingContent) #this dummy is never used. Just to split words of testing content.
                uniqueWordsFreqsTuple = uniqueWordsFreqsTuple\
                    [numWordsToDiscardTop:numWordsToDiscardTop+numWordsToConstructFeat[3]]
                listWords = [i for (i,j) in uniqueWordsFreqsTuple]
                printTOP10Words(listWords)

                #Afterwards we construct features from the training and testing
                #content. Specifically, these features set is a matrix of
                #size number of words considered times number of reviews in content.
                featuresTraining = constructFeatures(trainingContent, listWords)
                featuresTesting = constructFeatures(testingContent, listWords)

                #get prior and conditional distributions from training content
                prior = getPriorDist(trainingContent)
                classLabelsTr = getClassLabels(trainingContent)
                cpds = getCPDS(featuresTraining, classLabelsTr)

                #test the system using the probability distributions on the test set.
                predLabels = testSystem(featuresTesting,cpds,prior)
                classLabelsTs = getClassLabels(testingContent)

                #get the 0/1 loss given the predicted and actual class labels.
                zeroOneLoss = findZeroOneLoss(predLabels,classLabelsTs)
                print 'ZERO-ONE-LOSS ' + str(zeroOneLoss)
                zeroOneLossList.append(zeroOneLoss)

                baselineLabels = [1 for x in range(0,len(testingContent))]
                baselineLoss = findZeroOneLoss(baselineLabels,classLabelsTs)
                baslineLossList.append(baselineLoss)

            plt.plot(numTrials,zeroOneLossList)
            plt.plot(numTrials,baslineLossList,'r')
            plt.ylabel('Zero-one Loss')
            plt.xlabel('Number of Trial')
            plt.title('Zero One loss across 10 trials for ' + str(percentTraining[counterSizes]) +' % training set ')
            axes = plt.gca()
            axes.set_ylim([0,0.6])
            plt.savefig('ZeroOnelossacross10trialsfor' + str(percentTraining[counterSizes]) +'trainingset.png')
            plt.show()
            plt.close()

            avgLoss = average(zeroOneLossList)
            stdDev = np.std(zeroOneLossList)
            avgBaseLineLosses.append(average(baslineLossList))
            avgZeroOneLosses.append(avgLoss)
            print 'Average Zero One loss across 10 trials ' + str(avgLoss)
            print 'Std Deviation for Zero One loss across 10 trials ' + str(stdDev)

        plt.plot(trainingSetSize,avgZeroOneLosses)
        plt.plot(trainingSetSize,avgBaseLineLosses,'r')
        plt.ylabel('Zero-one Loss')
        plt.xlabel('Training Set Size')
        plt.title('Zero One loss for different training set sizes ')
        axes = plt.gca()
        axes.set_ylim([0,0.6])

        plt.savefig('acrossTrainingsets.png')
        plt.show()
        plt.close()


    if ExploreFeatureSpace:
        avgZeroOneLosses = []
        avgBaseLineLosses = []
        numOfFeats = []
        for counterNumFeats in range(0,len(numWordsToConstructFeat)):

            zeroOneLossList = []
            baslineLossList = []
            numTrials = []

            for trials in range(0,10):

                numTrials.append(trials)
                #from trainingFileName given as input read in the csv file and
                #store in csvContent variable.
                trainingFileName = arg[1]
                csvContent = readCSVFile(trainingFileName)

                #from all the csv content for the training and testing content
                #based on the random sampling mentioned in the handout.
                #this uses the percentTraining which is a global parameter
                #defining the percent of csvContent to be used for training.
                #random sampling is done for figuring out which reviews form
                #training set. The rest goes to testing set.
                numTrainingEx = int((percentTraining[4] / 100.0) * len(csvContent))
                listAllIdsContent = range(0,len(csvContent))
                randIdsTr = random.sample(listAllIdsContent, numTrainingEx)
                randIdsTs = list(set(listAllIdsContent)- set(randIdsTr))
                trainingContent = [csvContent[i] for i in randIdsTr]
                testingContent = [csvContent[i] for i in randIdsTs]

                #get the unique words from the training content.
                #we get a sorted list of word in a decreasing order wrt
                #the collection frequency. Top numWordsToDiscardTop words
                #are discarded as they are really high frequency and are
                #proabably stopwords. Then we consider only numWordsToConstructFeat
                #number of words. Afterwards we print the top 10 words.
                #If there is a tie in the top 10 words we print both.
                uniqueWordsFreqsTuple = getUniqueWords(trainingContent)
                dummy = getUniqueWords(testingContent) #this dummy is never used. Just to split words of testing content.
                uniqueWordsFreqsTuple = uniqueWordsFreqsTuple\
                    [numWordsToDiscardTop:numWordsToDiscardTop+numWordsToConstructFeat[counterNumFeats]]
                listWords = [i for (i,j) in uniqueWordsFreqsTuple]
                printTOP10Words(listWords)
                if trials == 0:
                    numOfFeats.append(numWordsToConstructFeat[counterNumFeats])

                #Afterwards we construct features from the training and testing
                #content. Specifically, these features set is a matrix of
                #size number of words considered times number of reviews in content.
                featuresTraining = constructFeatures(trainingContent, listWords)
                featuresTesting = constructFeatures(testingContent, listWords)

                #get prior and conditional distributions from training content
                prior = getPriorDist(trainingContent)
                classLabelsTr = getClassLabels(trainingContent)
                cpds = getCPDS(featuresTraining, classLabelsTr)

                #test the system using the probability distributions on the test set.
                predLabels = testSystem(featuresTesting,cpds,prior)
                classLabelsTs = getClassLabels(testingContent)

                #get the 0/1 loss given the predicted and actual class labels.
                zeroOneLoss = findZeroOneLoss(predLabels,classLabelsTs)
                print 'ZERO-ONE-LOSS ' + str(zeroOneLoss)
                zeroOneLossList.append(zeroOneLoss)

                baselineLabels = [1 for x in range(0,len(testingContent))]
                baselineLoss = findZeroOneLoss(baselineLabels,classLabelsTs)
                baslineLossList.append(baselineLoss)

            plt.plot(numTrials,zeroOneLossList)
            plt.plot(numTrials,baslineLossList,'r')
            plt.ylabel('Zero-one Loss')
            plt.xlabel('Number of Trial')
            plt.title('Zero One loss across 10 trials for ' + str(numWordsToConstructFeat[counterNumFeats]) +' number of features ')
            axes = plt.gca()
            axes.set_ylim([0,0.6])
            plt.savefig('ZeroOnelossacross10trialsfor' + str(numWordsToConstructFeat[counterNumFeats]) +'numberoffeatures.png')
            plt.show()
            plt.close()

            avgLoss = average(zeroOneLossList)
            stdDev = np.std(zeroOneLossList)
            avgBaseLineLosses.append(average(baslineLossList))
            avgZeroOneLosses.append(avgLoss)
            print 'Average Zero One loss across 10 trials ' + str(avgLoss)
            print 'Std Deviation for Zero One loss across 10 trials ' + str(stdDev)

        plt.plot(numOfFeats,avgZeroOneLosses)
        plt.plot(numOfFeats,avgBaseLineLosses,'r')
        plt.ylabel('Zero-one Loss')
        plt.xlabel('Number of Features')
        plt.title('Zero One loss for different feature set size ')
        axes = plt.gca()
        axes.set_ylim([0,0.6])

        plt.savefig('acrossNumberOfFeats.png')
        plt.show()
        plt.close()


    if UsualSetting:

        #from trainingFileName and testingfilename given as input read in the csv file and
        #store in variable.
        trainingFileName = arg[1]
        trainingContent = readCSVFile(trainingFileName)
        testingFileName = arg[2]
        testingContent = readCSVFile(testingFileName)

        #get the unique words from the training content.
        #we get a sorted list of word in a decreasing order wrt
        #the collection frequency. Top numWordsToDiscardTop words
        #are discarded as they are really high frequency and are
        #proabably stopwords. Then we consider only numWordsToConstructFeat
        #number of words. Afterwards we print the top 10 words.
        #If there is a tie in the top 10 words we print both.
        uniqueWordsFreqsTuple = getUniqueWords(trainingContent)
        dummy = getUniqueWords(testingContent) #this dummy is never used. Just to split words of testing content.
        uniqueWordsFreqsTuple = uniqueWordsFreqsTuple\
            [numWordsToDiscardTop:numWordsToDiscardTop+numWordsToConstructFeat[3]]
        listWords = [i for (i,j) in uniqueWordsFreqsTuple]
        printTOP10Words(listWords)

        #Afterwards we construct features from the training and testing
        #content. Specifically, these features set is a matrix of
        #size number of words considered times number of reviews in content.
        featuresTraining = constructFeatures(trainingContent, listWords)
        featuresTesting = constructFeatures(testingContent, listWords)

        #get prior and conditional distributions from training content
        prior = getPriorDist(trainingContent)
        classLabelsTr = getClassLabels(trainingContent)
        cpds = getCPDS(featuresTraining, classLabelsTr)

        #test the system using the probability distributions on the test set.
        predLabels = testSystem(featuresTesting,cpds,prior)
        classLabelsTs = getClassLabels(testingContent)

        #get the 0/1 loss given the predicted and actual class labels.
        zeroOneLoss = findZeroOneLoss(predLabels,classLabelsTs)
        print 'ZERO-ONE-LOSS ' + str(zeroOneLoss)


if __name__ == "__main__":
    main(sys.argv[0:])
