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
ExploreDataSpace = 0
ExploreFeatureSpace = 0
ExploreDepthSpace = 0
ExploreTreeSpace = 0

#Following are settings under which we want to run the
#system. If ExploreTrainingSetSize is 1 then percentTraining
#is used to explore different percentages of training size
#This value is set to 50% in case we are running ExploreFeatureSpace
#numWordsToDiscardTop is always 100.
#numWordsToConstructFeat is to explore the feature set space.
#It is set to a constant 500 when we want to run the system
#uunder ExploreTrainingSetSize setting.
percentTraining = [2.5, 5, 12.5, 25]
numWordsToDiscardTop = 100
numWordsToConstructFeat = 1000
numFolds=10
foldSize=200
featureSizeList = [200, 500, 1000, 1500]

#parameters of SVM
SVMLAMBDA = 0.01
SVMETA = 0.5
NUMITERS = 100
TOLERANCE = 1e-6

#parameters of DT
DTDEPTHLIMIT = 10
DTEXAMPLELIMIT = 10
DepthsList = [5, 10, 15, 20]

#parameter of BT
BTNUMTREES = 50

#parameter of RF
RFNUMTREES = 50

#parameter of BS
BSNUMTREES = 50

NumTreesList = [10,25,50,100]
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

#######################################DT FUNCS#################################

class Node():

    def __init__(self):
        self.featIdx = None
        self.lChild = None
        self.rChild = None
        self.terminalClassL = None
        self.terminalClassR = None

def prepareDataDT(content, words):
    data = np.zeros((len(content), len(words) + 1), dtype=float)

    for i in range(0, len(content)):
        for j in range(0, len(words)):
            if words[j] in content[i][2]:
                data[i][j] = 1.0
            else:
                data[i][j] = 0.0
        data[i][j+1] = float(content[i][1])
    return data

def divideData(data):

    attrIdxDivide = 0
    attrGGDivide = 1000
    attrEgsDivided = None
    for i in range(0,len(data[0])-1):
        for j in range(0,len(data)):
            egsEqualAttr = []
            egsDiffAttr = []

            for k in range(0,len(data)):
                if data[k][i] == data[j][i]:
                    egsEqualAttr.append(k)
                elif data[k][i] != data[j][i]:
                    egsDiffAttr.append(k)

        count0classesinegsEqualAttr = 0.0
        count1classesinegsEqualAttr = 0.0
        for m in range(0,len(egsEqualAttr)):
            if data[egsEqualAttr[m]][len(data[0])-1] == 0.0:
                count0classesinegsEqualAttr += 1.0
            elif data[egsEqualAttr[m]][len(data[0])-1] == 1.0:
                count1classesinegsEqualAttr += 1.0
        count0classesinegsEqualAttr = count0classesinegsEqualAttr / len(egsEqualAttr)
        count1classesinegsEqualAttr = count1classesinegsEqualAttr / len(egsEqualAttr)

        count0classesinegsDiffAttr = 0.0
        count1classesinegsDiffAttr = 0.0
        for m in range(0,len(egsDiffAttr)):
            if data[egsDiffAttr[m]][len(data[0])-1] == 0.0:
                count0classesinegsDiffAttr += 1.0
            elif data[egsDiffAttr[m]][len(data[0])-1] == 1.0:
                count1classesinegsDiffAttr += 1.0
        count0classesinegsDiffAttr = count0classesinegsDiffAttr / len(egsDiffAttr)
        count1classesinegsDiffAttr = count1classesinegsDiffAttr / len(egsDiffAttr)

        GG = count0classesinegsEqualAttr * (1.0 - count0classesinegsEqualAttr) + \
             count1classesinegsEqualAttr * (1.0 - count1classesinegsEqualAttr) + \
             count0classesinegsDiffAttr  * (1.0 - count0classesinegsDiffAttr ) + \
             count1classesinegsDiffAttr  * (1.0 - count1classesinegsDiffAttr)

        if GG < attrGGDivide:
            attrIdxDivide = i
            attrGGDivide = GG
            node = Node
            node.featIdx = attrIdxDivide
            node.lChild = egsEqualAttr
            node.rChild = egsDiffAttr

    return node

def splitData(features, classes, isRandFeats):
    # print "features start of splitData: "
    # print features
    # print "classes start of splitData: "
    # print classes

    if isRandFeats:
        numDownsampleFeats = int(np.sqrt(len(features)))
        allFeats = np.arange(0,len(features))
        randomFeats = np.random.choice(allFeats,numDownsampleFeats,replace=False)
        # print len(features), len(features[0])
        features = features[randomFeats]
        # print len(features), len(features[0])

    numFeats = len(features)
    numExamples = len(features[0])

    num1classes = float(np.count_nonzero(classes))
    num0classes = float(numExamples) - num1classes
    GiniS = 1.0 - np.square((num1classes/float(numExamples))) \
                - np.square((num0classes/float(numExamples)))
    bestGG = 0.0
    bestFeatIdx = 0

    for countFeat in range(0, numFeats):
        num1feats = float(np.count_nonzero(features[countFeat]))
        num0feats = float(numExamples) - num1feats

        num0classes0feats = float(len(np.intersect1d(np.where(classes==0.0),\
                                           np.where(features[countFeat]==0.0))))
        num0classes1feats = float(len(np.intersect1d(np.where(classes==0.0),\
                                           np.where(features[countFeat]==1.0))))
        num1classes0feats = float(len(np.intersect1d(np.where(classes==1.0),\
                                           np.where(features[countFeat]==0.0))))
        num1classes1feats = float(len(np.intersect1d(np.where(classes==1.0),\
                                           np.where(features[countFeat]==1.0))))

        if num1feats != 0:
            Ginifeats1 = 1.0 - np.square(num1classes1feats/num1feats) \
                             - np.square(num0classes1feats/num1feats)
            # Ginifeats1 = (num1classes1feats/num1feats) * (num1classes1feats/num1feats)\
            #             +(num0classes1feats/num1feats) * (num0classes1feats/num1feats)
        else:
            Ginifeats1 = 0.0


        if num0feats != 0:
            Ginifeats0 = 1.0 - np.square(num1classes0feats/num0feats) \
                             - np.square(num0classes0feats/num0feats)
            # Ginifeats0 = (num1classes0feats/num0feats) * (num1classes0feats/num0feats)\
            #             +(num0classes0feats/num0feats) * (num0classes0feats/num0feats)
        else:
            Ginifeats0 = 0.0


        GG = GiniS - (num1feats/float(numExamples)) * Ginifeats1 \
                   - (num0feats/float(numExamples)) * Ginifeats0
        # GG = (num1feats/float(numExamples)) * Ginifeats1 + (num0feats/float(numExamples)) * Ginifeats0
        if isRandFeats:
            if GG >= bestGG:
                bestGG = GG
                bestFeatIdx = randomFeats[countFeat]
                leftChild = np.asarray(np.where(features[countFeat]==0.0))
                rightChild = np.asarray(np.where(features[countFeat]==1.0))
        else:
            if GG >= bestGG:
                bestGG = GG
                bestFeatIdx = countFeat
                leftChild = np.asarray(np.where(features[countFeat]==0.0))
                rightChild = np.asarray(np.where(features[countFeat]==1.0))

    node = Node()
    node.featIdx = bestFeatIdx
    node.lChild = leftChild[0]
    node.rChild = rightChild[0]
    # print "end of split data"
    # print "feature id, len left child, len right child"
    # print node.featIdx, len(node.lChild), len(node.rChild)
    # print "left child:"
    # print node.lChild
    # print "right child:"
    # print node.rChild
    # print "left child classes:"
    # print classes[node.lChild]
    # print "right child classes:"
    # print classes[node.rChild]
    return node

def findMajorityClass(classes):
    num1s = 0
    num0s = 0
    for c in classes:
        if c == 1.0:
            num1s += 1
        elif c == 0.0:
            num0s += 1

    if num1s >= num0s:
        return 1.0
    else:
        return 0.0

def buildTree(node, features, classes, currDepth, isRandFeats,DepthLim):
    # print "depth is " + str(currDepth)

    l = node.lChild
    r = node.rChild
    numEgsL = len(l)
    numEgsR = len(r)

    if numEgsL==0 or numEgsR==0:
        if numEgsL == 0:
            combinedLR = r
        if numEgsR == 0:
            combinedLR = l
        node.terminalClassL = findMajorityClass(classes[combinedLR])
        node.terminalClassR = findMajorityClass(classes[combinedLR])
        # print "left and right are: " + str(node.terminalClassL)
        node.lChild = None
        node.rChild = None
        return

    if currDepth >= DepthLim:
        node.terminalClassL = findMajorityClass(classes[l])
        node.terminalClassR = findMajorityClass(classes[r])
        # print "left is: " + str(node.terminalClassL) + "depthlimit"
        # print "right is: " + str(node.terminalClassR) + "depthlimit"
        node.lChild = None
        node.rChild = None
        return

    if numEgsL <= DTEXAMPLELIMIT:
        node.terminalClassL = findMajorityClass(classes[l])
        # print "left is: " + str(node.terminalClassL) + "examplelimit"
        node.lChild = None
    else:
        allIdsEgs = np.arange(len(features[0]))
        remainingEgs = np.setdiff1d(allIdsEgs,r)
        tx =  features[:,remainingEgs]
        node.lChild = splitData(tx,classes[remainingEgs],isRandFeats)
        buildTree(node.lChild,tx, classes[remainingEgs], currDepth+1, isRandFeats,DepthLim)

    if numEgsR <= DTEXAMPLELIMIT:
        node.terminalClassR = findMajorityClass(classes[r])
        # print "right is: " + str(node.terminalClassR) + " examplelimit"
        node.rChild = None
    else:
        allIdsEgs = np.arange(len(features[0]))
        remainingEgs = np.setdiff1d(allIdsEgs,l)
        tx =  features[:,remainingEgs]
        node.rChild = splitData(tx,classes[remainingEgs], isRandFeats)
        buildTree(node.rChild, tx, classes[remainingEgs], currDepth+1, isRandFeats,DepthLim)

def predictSingleEg(data, node):
    # print node.featIdx
    if data[node.featIdx] == 0.0:
        if node.lChild is not None:
            return predictSingleEg(data, node.lChild)
        else:
            return node.terminalClassL
    elif data[node.featIdx] == 1.0:
        if node.rChild is not None:
            return predictSingleEg(data, node.rChild)
        else:
            return node.terminalClassR

def predictUsingDT(testData, node):

    predLabels = np.zeros((len(testData)),dtype=float)
    for i in range(0,len(testData)):
        predClassEg = predictSingleEg(testData[i,:-1],node)
        predLabels[i] = predClassEg
    return predLabels

#######################################BT FUNCS##################################
def finalPredsBT(predsALLTrees):
    finalPreds = []
    for countEg in range(0,len(predsALLTrees)):
        classes = [predsALLTrees[i][countEg] for i in range(0,len(predsALLTrees))]
        majorClass = findMajorityClass(classes)
        finalPreds.append(majorClass)
    return np.asarray(finalPreds, dtype=float)

#######################################BS FUNCS##################################
def findError(pred, true, w):
    err = 0.0
    for i in range(0, len(pred)):
        if pred[i] != true[i]:
            err += w[i]
    return err

def updateWeightsBS(w, a, pred, true):
    updatedW = w
    totalW = 0.0
    for i in range(0,len(w)):
        if pred[i] != true[i]:
            updatedW[i] = w[i] * np.exp(a)
        else:
            updatedW[i] = w[i]
        totalW += updatedW[i]
    return updatedW / totalW

def findfinalClassBS(c,a):
    finalClass = 0.0
    for i in range(0,len(c)):
        finalClass += a[i] * c[i]
    if finalClass > 0.0:
        return 1.0
    else:
        return 0.0

def finalPredsBS(predsALLTrees, a):

    finalPreds = []
    for countEg in range(0,len(predsALLTrees)):
        classes = [predsALLTrees[i][countEg] for i in range(0,len(predsALLTrees))]
        classes = np.asarray(classes,dtype=float)
        classes[classes==0.0] = -1.0
        finalClass = findfinalClassBS(classes,a)
        finalPreds.append(finalClass)
    return np.asarray(finalPreds, dtype=float)

#######################################SVM FUNCS#################################

def getFeatValuesForEx(features, countEx):
    return np.array([float(l[countEx]) for l in features])

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

    if ExploreDataSpace:

        trainingFileName = arg[1]
        csvContent = readCSVFile(trainingFileName)
        randIDLists = getRandomIDList(len(csvContent))

        for countRevs in range(0,len(csvContent)):
            csvContent[countRevs][2] = csvContent[countRevs][2].lower().translate(None, string.punctuation).split()

        avgLSVMs = []
        seLSVMs = []
        avgLDTs = []
        seLDTs = []
        avgLBTs = []
        seLBTs = []
        avgLBSs = []
        seLBSs = []
        avgLRFs = []
        seLRFs = []
        TSS = []
        for counterSizes in range(0,len(percentTraining)):
            print "======================================="
            LDT = []
            LSVM = []
            LBT = []
            LBS = []
            LRF = []
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
                uniqueWordsFreqsTuple = uniqueWordsFreqsTuple\
                    [numWordsToDiscardTop:numWordsToDiscardTop+numWordsToConstructFeat]
                listWords = [i for (i,j) in uniqueWordsFreqsTuple]

                featuresTraining = constructFeatures(trainingContent, listWords)
                featuresTraining = featuresTraining.astype(float)

                featuresTesting = constructFeatures(testingContent, listWords)
                featuresTesting = featuresTesting.astype(float)

                featuresTestingDT = prepareDataDT(testingContent, listWords)

                classLabelsTr = getClassLabels(trainingContent)
                classLabelsTr = classLabelsTr.astype(float)

                classLabelsTs = getClassLabels(testingContent)
                classLabelsTs = classLabelsTs.astype(float)

                rootNode = splitData(featuresTraining, classLabelsTr, 0)
                buildTree(rootNode, featuresTraining, classLabelsTr, 1, 0,DTDEPTHLIMIT)

                predictLabelsDT = predictUsingDT(featuresTestingDT, rootNode)

                zeroOneLossDT = findZeroOneLoss(predictLabelsDT,classLabelsTs)
                print 'ZERO-ONE-LOSS-DT ' + str(zeroOneLossDT)
                LDT.append(zeroOneLossDT)

                roots = []
                for countTree in range(0,BTNUMTREES):
                    randIdsEgs = np.random.choice(range(0,len(trainingContent)),len(trainingContent),replace=True)
                    temp = []
                    for i in range(0,len(randIdsEgs)):
                        temp.append(trainingContent[randIdsEgs[i]])

                    featuresTraining = constructFeatures(temp, listWords)
                    classLabelsTr = getClassLabels(temp)

                    rootNode = splitData(featuresTraining, classLabelsTr, 0)
                    buildTree(rootNode, featuresTraining, classLabelsTr, 1, 0, DTDEPTHLIMIT)

                    roots.append(rootNode)

                featuresTesting = prepareDataDT(testingContent,listWords)
                classLabelsTs = getClassLabels(testingContent)
                classLabelsTs = classLabelsTs.astype(float)

                predsAllTrees = []
                for j in range(0,BTNUMTREES):
                    predictLabelsDT = predictUsingDT(featuresTesting, roots[j])
                    predsAllTrees.append(predictLabelsDT)
                finalPreds = finalPredsBT(predsAllTrees)
                zeroOneLossBT = findZeroOneLoss(finalPreds,classLabelsTs)
                print 'ZERO-ONE-LOSS-BT ' + str(zeroOneLossBT)
                LBT.append(zeroOneLossBT)

                roots = []
                for countTree in range(0,RFNUMTREES):
                    randIdsEgs = np.random.choice(range(0,len(trainingContent)),len(trainingContent),replace=True)
                    temp = []
                    for i in range(0,len(randIdsEgs)):
                        temp.append(trainingContent[randIdsEgs[i]])

                    featuresTraining = constructFeatures(temp, listWords)
                    classLabelsTr = getClassLabels(temp)

                    rootNode = splitData(featuresTraining, classLabelsTr, 1)
                    buildTree(rootNode, featuresTraining, classLabelsTr, 1, 1,DTDEPTHLIMIT)

                    roots.append(rootNode)

                featuresTesting = prepareDataDT(testingContent,listWords)
                classLabelsTs = getClassLabels(testingContent)
                classLabelsTs = classLabelsTs.astype(float)

                predsAllTrees = []
                for j in range(0,RFNUMTREES):
                    predictLabelsDT = predictUsingDT(featuresTesting, roots[j])
                    predsAllTrees.append(predictLabelsDT)
                finalPreds = finalPredsBT(predsAllTrees)
                zeroOneLossRF = findZeroOneLoss(finalPreds,classLabelsTs)
                print 'ZERO-ONE-LOSS-RF ' + str(zeroOneLossRF)
                LRF.append(zeroOneLossRF)

                roots = []
                am = []
                ws = np.ones((len(trainingContent)),dtype=float) / float(len(trainingContent))
                featuresTrainingAlldata = prepareDataDT(trainingContent, listWords)
                classLabelsTrAlldata = getClassLabels(trainingContent)

                allEgs = np.arange(0,len(trainingContent))
                for countTree in range(0,BSNUMTREES):
                    randomEgs = np.random.choice(allEgs,len(trainingContent), replace=True, p = ws)

                    trainData = []
                    for cEg in range(0, len(randomEgs)):
                        trainData.append(trainingContent[randomEgs[cEg]])

                    featuresTraining = constructFeatures(trainData, listWords)
                    classLabelsTr = getClassLabels(trainData)

                    rootNode = splitData(featuresTraining, classLabelsTr, 0)
                    buildTree(rootNode, featuresTraining, classLabelsTr, 1, 0,DTDEPTHLIMIT)
                    roots.append(rootNode)

                    classLabelsTrPlusMinus = classLabelsTrAlldata
                    classLabelsTrPlusMinus[classLabelsTrPlusMinus==0.0] = -1.0

                    predictLabelsDT = predictUsingDT(featuresTrainingAlldata, rootNode)
                    predictLabelsDTPlusMinus = predictLabelsDT
                    predictLabelsDTPlusMinus[predictLabelsDTPlusMinus==0.0] = -1.0

                    err = findError(predictLabelsDTPlusMinus, classLabelsTrPlusMinus, ws)
                    alpha = 0.5 * np.log((1-err)/err)
                    am.append(alpha)

                    ws = updateWeightsBS(ws, alpha, predictLabelsDTPlusMinus, classLabelsTrPlusMinus)

                featuresTesting = prepareDataDT(testingContent,listWords)
                classLabelsTs = getClassLabels(testingContent)
                classLabelsTs = classLabelsTs.astype(float)

                predsAllTrees = []
                for j in range(0,BSNUMTREES):
                    predictLabelsDT = predictUsingDT(featuresTesting, roots[j])
                    predictLabelsDT[predictLabelsDT==0.0] = -1.0
                    predsAllTrees.append(predictLabelsDT)
                finalPreds = finalPredsBS(predsAllTrees,am)
                zeroOneLossBS = findZeroOneLoss(finalPreds,classLabelsTs)
                print 'ZERO-ONE-LOSS-BS ' + str(zeroOneLossBS)
                LBS.append(zeroOneLossBS)

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
                LSVM.append(zeroOneLossSVM)

            avgLSVM = average(LSVM)
            avgLSVMs.append(avgLSVM)

            stdLSVM = np.std(LSVM)
            seLSVM = stdLSVM / float(numFolds)

            seLSVMs.append(seLSVM)

            avgLDT = average(LDT)
            avgLDTs.append(avgLDT)

            stdLDT = np.std(LDT)
            seLDT = stdLDT / float(numFolds)

            seLDTs.append(seLDT)

            avgLBT = average(LBT)
            avgLBTs.append(avgLBT)

            stdLBT = np.std(LBT)
            seLBT = stdLBT / float(numFolds)

            seLBTs.append(seLBT)

            avgLBS = average(LBS)
            avgLBSs.append(avgLBS)

            stdLBS = np.std(LBS)
            seLBS = stdLBS / float(numFolds)

            seLBSs.append(seLBS)

            avgLRF = average(LRF)
            avgLRFs.append(avgLRF)

            stdLRF = np.std(LRF)
            seLRF = stdLRF / float(numFolds)

            seLRFs.append(seLRF)

        print "=======SVM======"
        print avgLSVMs
        print seLSVMs

        plt.figure()
        plt.errorbar(TSS, avgLSVMs, seLSVMs, label='SVM')
        plt.errorbar(TSS, avgLDTs, seLDTs, label='DT')
        plt.errorbar(TSS, avgLBTs, seLBTs, label='BT')
        plt.errorbar(TSS, avgLBSs, seLBSs, label='BS')
        plt.errorbar(TSS, avgLRFs, seLRFs, label='RF')
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

        avgLSVMs = []
        seLSVMs = []
        avgLDTs = []
        seLDTs = []
        avgLBTs = []
        seLBTs = []
        avgLBSs = []
        seLBSs = []
        avgLRFs = []
        seLRFs = []
        FSS = []
        for counterSizes in range(0,len(featureSizeList)):
            print "======================================="
            LDT = []
            LSVM = []
            LBT = []
            LBS = []
            LRF = []
            for fold in range(0,numFolds):

                testSetIDs = randIDLists[fold]
                restLists = randIDLists[0:fold]+randIDLists[fold+1:]
                restIDs = [item for sublist in restLists for item in sublist]

                numTrainingEx = int((percentTraining[3] / 100.0) * len(csvContent))
                randIdsTr = random.sample(restIDs, numTrainingEx)

                if fold == 0:
                    FSS.append(featureSizeList[counterSizes])

                trainingContent = []
                testingContent = []

                for i in range(0,len(randIdsTr)):
                    temp = csvContent[randIdsTr[i]]
                    trainingContent.append(temp)
                for i in range(0,len(testSetIDs)):
                    temp = csvContent[testSetIDs[i]]
                    testingContent.append(temp)

                uniqueWordsFreqsTuple = getUniqueWords(trainingContent)
                uniqueWordsFreqsTuple = uniqueWordsFreqsTuple\
                    [numWordsToDiscardTop:numWordsToDiscardTop+featureSizeList[counterSizes]]
                listWords = [i for (i,j) in uniqueWordsFreqsTuple]

                featuresTraining = constructFeatures(trainingContent, listWords)
                featuresTraining = featuresTraining.astype(float)

                featuresTesting = constructFeatures(testingContent, listWords)
                featuresTesting = featuresTesting.astype(float)

                featuresTestingDT = prepareDataDT(testingContent, listWords)

                classLabelsTr = getClassLabels(trainingContent)
                classLabelsTr = classLabelsTr.astype(float)

                classLabelsTs = getClassLabels(testingContent)
                classLabelsTs = classLabelsTs.astype(float)

                rootNode = splitData(featuresTraining, classLabelsTr, 0)
                buildTree(rootNode, featuresTraining, classLabelsTr, 1, 0,DTDEPTHLIMIT)

                predictLabelsDT = predictUsingDT(featuresTestingDT, rootNode)

                zeroOneLossDT = findZeroOneLoss(predictLabelsDT,classLabelsTs)
                print 'ZERO-ONE-LOSS-DT ' + str(zeroOneLossDT)
                LDT.append(zeroOneLossDT)

                roots = []
                for countTree in range(0,BTNUMTREES):
                    randIdsEgs = np.random.choice(range(0,len(trainingContent)),len(trainingContent),replace=True)
                    temp = []
                    for i in range(0,len(randIdsEgs)):
                        temp.append(trainingContent[randIdsEgs[i]])

                    featuresTraining = constructFeatures(temp, listWords)
                    classLabelsTr = getClassLabels(temp)

                    rootNode = splitData(featuresTraining, classLabelsTr, 0)
                    buildTree(rootNode, featuresTraining, classLabelsTr, 1, 0,DTDEPTHLIMIT)

                    roots.append(rootNode)

                featuresTesting = prepareDataDT(testingContent,listWords)
                classLabelsTs = getClassLabels(testingContent)
                classLabelsTs = classLabelsTs.astype(float)

                predsAllTrees = []
                for j in range(0,BTNUMTREES):
                    predictLabelsDT = predictUsingDT(featuresTesting, roots[j])
                    predsAllTrees.append(predictLabelsDT)
                finalPreds = finalPredsBT(predsAllTrees)
                zeroOneLossBT = findZeroOneLoss(finalPreds,classLabelsTs)
                print 'ZERO-ONE-LOSS-BT ' + str(zeroOneLossBT)
                LBT.append(zeroOneLossBT)

                roots = []
                for countTree in range(0,RFNUMTREES):
                    randIdsEgs = np.random.choice(range(0,len(trainingContent)),len(trainingContent),replace=True)
                    temp = []
                    for i in range(0,len(randIdsEgs)):
                        temp.append(trainingContent[randIdsEgs[i]])

                    featuresTraining = constructFeatures(temp, listWords)
                    classLabelsTr = getClassLabels(temp)

                    rootNode = splitData(featuresTraining, classLabelsTr, 1)
                    buildTree(rootNode, featuresTraining, classLabelsTr, 1, 1,DTDEPTHLIMIT)

                    roots.append(rootNode)

                featuresTesting = prepareDataDT(testingContent,listWords)
                classLabelsTs = getClassLabels(testingContent)
                classLabelsTs = classLabelsTs.astype(float)

                predsAllTrees = []
                for j in range(0,RFNUMTREES):
                    predictLabelsDT = predictUsingDT(featuresTesting, roots[j])
                    predsAllTrees.append(predictLabelsDT)
                finalPreds = finalPredsBT(predsAllTrees)
                zeroOneLossRF = findZeroOneLoss(finalPreds,classLabelsTs)
                print 'ZERO-ONE-LOSS-RF ' + str(zeroOneLossRF)
                LRF.append(zeroOneLossRF)

                roots = []
                am = []
                ws = np.ones((len(trainingContent)),dtype=float) / float(len(trainingContent))
                featuresTrainingAlldata = prepareDataDT(trainingContent, listWords)
                classLabelsTrAlldata = getClassLabels(trainingContent)

                allEgs = np.arange(0,len(trainingContent))
                for countTree in range(0,BSNUMTREES):
                    randomEgs = np.random.choice(allEgs,len(trainingContent), replace=True, p = ws)

                    trainData = []
                    for cEg in range(0, len(randomEgs)):
                        trainData.append(trainingContent[randomEgs[cEg]])

                    featuresTraining = constructFeatures(trainData, listWords)
                    classLabelsTr = getClassLabels(trainData)

                    rootNode = splitData(featuresTraining, classLabelsTr, 0)
                    buildTree(rootNode, featuresTraining, classLabelsTr, 1, 0,DTDEPTHLIMIT)
                    roots.append(rootNode)

                    classLabelsTrPlusMinus = classLabelsTrAlldata
                    classLabelsTrPlusMinus[classLabelsTrPlusMinus==0.0] = -1.0

                    predictLabelsDT = predictUsingDT(featuresTrainingAlldata, rootNode)
                    predictLabelsDTPlusMinus = predictLabelsDT
                    predictLabelsDTPlusMinus[predictLabelsDTPlusMinus==0.0] = -1.0

                    err = findError(predictLabelsDTPlusMinus, classLabelsTrPlusMinus, ws)
                    alpha = 0.5 * np.log((1-err)/err)
                    am.append(alpha)

                    ws = updateWeightsBS(ws, alpha, predictLabelsDTPlusMinus, classLabelsTrPlusMinus)

                featuresTesting = prepareDataDT(testingContent,listWords)
                classLabelsTs = getClassLabels(testingContent)
                classLabelsTs = classLabelsTs.astype(float)

                predsAllTrees = []
                for j in range(0,BSNUMTREES):
                    predictLabelsDT = predictUsingDT(featuresTesting, roots[j])
                    predictLabelsDT[predictLabelsDT==0.0] = -1.0
                    predsAllTrees.append(predictLabelsDT)
                finalPreds = finalPredsBS(predsAllTrees,am)
                zeroOneLossBS = findZeroOneLoss(finalPreds,classLabelsTs)
                print 'ZERO-ONE-LOSS-BS ' + str(zeroOneLossBS)
                LBS.append(zeroOneLossBS)

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
                LSVM.append(zeroOneLossSVM)

            avgLSVM = average(LSVM)
            avgLSVMs.append(avgLSVM)

            stdLSVM = np.std(LSVM)
            seLSVM = stdLSVM / float(numFolds)

            seLSVMs.append(seLSVM)

            avgLDT = average(LDT)
            avgLDTs.append(avgLDT)

            stdLDT = np.std(LDT)
            seLDT = stdLDT / float(numFolds)

            seLDTs.append(seLDT)

            avgLBT = average(LBT)
            avgLBTs.append(avgLBT)

            stdLBT = np.std(LBT)
            seLBT = stdLBT / float(numFolds)

            seLBTs.append(seLBT)

            avgLBS = average(LBS)
            avgLBSs.append(avgLBS)

            stdLBS = np.std(LBS)
            seLBS = stdLBS / float(numFolds)

            seLBSs.append(seLBS)

            avgLRF = average(LRF)
            avgLRFs.append(avgLRF)

            stdLRF = np.std(LRF)
            seLRF = stdLRF / float(numFolds)

            seLRFs.append(seLRF)

        print "=======SVM======"
        print avgLSVMs
        print seLSVMs
        print "=======DT======"
        print avgLDTs
        print seLDTs
        print "=======BT======"
        print avgLBTs
        print seLBTs
        print "=======RF======"
        print avgLRFs
        print seLRFs
        print "=======BS======"
        print avgLBSs
        print seLBSs

        plt.figure()
        plt.errorbar(FSS, avgLSVMs, seLSVMs, label='SVM')
        plt.errorbar(FSS, avgLDTs, seLDTs, label='DT')
        plt.errorbar(FSS, avgLBTs, seLBTs, label='BT')
        plt.errorbar(FSS, avgLBSs, seLBSs, label='BS')
        plt.errorbar(FSS, avgLRFs, seLRFs, label='RF')
        plt.ylabel('Zero-one Loss')
        plt.xlabel('Feature set sizes')
        plt.title('Zero One loss for different feature set size ')
        plt.legend()
        axes = plt.gca()
        axes.set_ylim([0,0.6])
        plt.savefig('Analysis2.png')
        plt.show()
        plt.close()

    if ExploreDepthSpace:

        trainingFileName = arg[1]
        csvContent = readCSVFile(trainingFileName)
        randIDLists = getRandomIDList(len(csvContent))

        for countRevs in range(0,len(csvContent)):
            csvContent[countRevs][2] = csvContent[countRevs][2].lower().translate(None, string.punctuation).split()

        avgLSVMs = []
        seLSVMs = []
        avgLDTs = []
        seLDTs = []
        avgLBTs = []
        seLBTs = []
        avgLBSs = []
        seLBSs = []
        avgLRFs = []
        seLRFs = []
        DSS = []
        for counterDepths in range(0,len(DepthsList)):
            print "======================================="
            LDT = []
            LSVM = []
            LBT = []
            LBS = []
            LRF = []
            for fold in range(0,numFolds):

                testSetIDs = randIDLists[fold]
                restLists = randIDLists[0:fold]+randIDLists[fold+1:]
                restIDs = [item for sublist in restLists for item in sublist]

                numTrainingEx = int((percentTraining[3] / 100.0) * len(csvContent))
                randIdsTr = random.sample(restIDs, numTrainingEx)

                if fold == 0:
                    DSS.append(DepthsList[counterDepths])

                trainingContent = []
                testingContent = []

                for i in range(0,len(randIdsTr)):
                    temp = csvContent[randIdsTr[i]]
                    trainingContent.append(temp)
                for i in range(0,len(testSetIDs)):
                    temp = csvContent[testSetIDs[i]]
                    testingContent.append(temp)

                uniqueWordsFreqsTuple = getUniqueWords(trainingContent)
                uniqueWordsFreqsTuple = uniqueWordsFreqsTuple\
                    [numWordsToDiscardTop:numWordsToDiscardTop+featureSizeList[2]]
                listWords = [i for (i,j) in uniqueWordsFreqsTuple]

                featuresTraining = constructFeatures(trainingContent, listWords)
                featuresTraining = featuresTraining.astype(float)

                featuresTesting = constructFeatures(testingContent, listWords)
                featuresTesting = featuresTesting.astype(float)

                featuresTestingDT = prepareDataDT(testingContent, listWords)

                classLabelsTr = getClassLabels(trainingContent)
                classLabelsTr = classLabelsTr.astype(float)

                classLabelsTs = getClassLabels(testingContent)
                classLabelsTs = classLabelsTs.astype(float)

                rootNode = splitData(featuresTraining, classLabelsTr, 0)
                buildTree(rootNode, featuresTraining, classLabelsTr, 1, 0,DepthsList[counterDepths])

                predictLabelsDT = predictUsingDT(featuresTestingDT, rootNode)

                zeroOneLossDT = findZeroOneLoss(predictLabelsDT,classLabelsTs)
                print 'ZERO-ONE-LOSS-DT ' + str(zeroOneLossDT)
                LDT.append(zeroOneLossDT)

                roots = []
                for countTree in range(0,BTNUMTREES):
                    randIdsEgs = np.random.choice(range(0,len(trainingContent)),len(trainingContent),replace=True)
                    temp = []
                    for i in range(0,len(randIdsEgs)):
                        temp.append(trainingContent[randIdsEgs[i]])

                    featuresTraining = constructFeatures(temp, listWords)
                    classLabelsTr = getClassLabels(temp)

                    rootNode = splitData(featuresTraining, classLabelsTr, 0)
                    buildTree(rootNode, featuresTraining, classLabelsTr, 1, 0,DepthsList[counterDepths])

                    roots.append(rootNode)

                featuresTesting = prepareDataDT(testingContent,listWords)
                classLabelsTs = getClassLabels(testingContent)
                classLabelsTs = classLabelsTs.astype(float)

                predsAllTrees = []
                for j in range(0,BTNUMTREES):
                    predictLabelsDT = predictUsingDT(featuresTesting, roots[j])
                    predsAllTrees.append(predictLabelsDT)
                finalPreds = finalPredsBT(predsAllTrees)
                zeroOneLossBT = findZeroOneLoss(finalPreds,classLabelsTs)
                print 'ZERO-ONE-LOSS-BT ' + str(zeroOneLossBT)
                LBT.append(zeroOneLossBT)

                roots = []
                for countTree in range(0,RFNUMTREES):
                    randIdsEgs = np.random.choice(range(0,len(trainingContent)),len(trainingContent),replace=True)
                    temp = []
                    for i in range(0,len(randIdsEgs)):
                        temp.append(trainingContent[randIdsEgs[i]])

                    featuresTraining = constructFeatures(temp, listWords)
                    classLabelsTr = getClassLabels(temp)

                    rootNode = splitData(featuresTraining, classLabelsTr, 1)
                    buildTree(rootNode, featuresTraining, classLabelsTr, 1, 1,DepthsList[counterDepths])

                    roots.append(rootNode)

                featuresTesting = prepareDataDT(testingContent,listWords)
                classLabelsTs = getClassLabels(testingContent)
                classLabelsTs = classLabelsTs.astype(float)

                predsAllTrees = []
                for j in range(0,RFNUMTREES):
                    predictLabelsDT = predictUsingDT(featuresTesting, roots[j])
                    predsAllTrees.append(predictLabelsDT)
                finalPreds = finalPredsBT(predsAllTrees)
                zeroOneLossRF = findZeroOneLoss(finalPreds,classLabelsTs)
                print 'ZERO-ONE-LOSS-RF ' + str(zeroOneLossRF)
                LRF.append(zeroOneLossRF)

                roots = []
                am = []
                ws = np.ones((len(trainingContent)),dtype=float) / float(len(trainingContent))
                featuresTrainingAlldata = prepareDataDT(trainingContent, listWords)
                classLabelsTrAlldata = getClassLabels(trainingContent)

                allEgs = np.arange(0,len(trainingContent))
                for countTree in range(0,BSNUMTREES):
                    randomEgs = np.random.choice(allEgs,len(trainingContent), replace=True, p = ws)

                    trainData = []
                    for cEg in range(0, len(randomEgs)):
                        trainData.append(trainingContent[randomEgs[cEg]])

                    featuresTraining = constructFeatures(trainData, listWords)
                    classLabelsTr = getClassLabels(trainData)

                    rootNode = splitData(featuresTraining, classLabelsTr, 0)
                    buildTree(rootNode, featuresTraining, classLabelsTr, 1, 0,DepthsList[counterDepths])
                    roots.append(rootNode)

                    classLabelsTrPlusMinus = classLabelsTrAlldata
                    classLabelsTrPlusMinus[classLabelsTrPlusMinus==0.0] = -1.0

                    predictLabelsDT = predictUsingDT(featuresTrainingAlldata, rootNode)
                    predictLabelsDTPlusMinus = predictLabelsDT
                    predictLabelsDTPlusMinus[predictLabelsDTPlusMinus==0.0] = -1.0

                    err = findError(predictLabelsDTPlusMinus, classLabelsTrPlusMinus, ws)
                    alpha = 0.5 * np.log((1-err)/err)
                    am.append(alpha)

                    ws = updateWeightsBS(ws, alpha, predictLabelsDTPlusMinus, classLabelsTrPlusMinus)

                featuresTesting = prepareDataDT(testingContent,listWords)
                classLabelsTs = getClassLabels(testingContent)
                classLabelsTs = classLabelsTs.astype(float)

                predsAllTrees = []
                for j in range(0,BSNUMTREES):
                    predictLabelsDT = predictUsingDT(featuresTesting, roots[j])
                    predictLabelsDT[predictLabelsDT==0.0] = -1.0
                    predsAllTrees.append(predictLabelsDT)
                finalPreds = finalPredsBS(predsAllTrees,am)
                zeroOneLossBS = findZeroOneLoss(finalPreds,classLabelsTs)
                print 'ZERO-ONE-LOSS-BS ' + str(zeroOneLossBS)
                LBS.append(zeroOneLossBS)

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
                LSVM.append(zeroOneLossSVM)

            avgLSVM = average(LSVM)
            avgLSVMs.append(avgLSVM)

            stdLSVM = np.std(LSVM)
            seLSVM = stdLSVM / float(numFolds)

            seLSVMs.append(seLSVM)

            avgLDT = average(LDT)
            avgLDTs.append(avgLDT)

            stdLDT = np.std(LDT)
            seLDT = stdLDT / float(numFolds)

            seLDTs.append(seLDT)

            avgLBT = average(LBT)
            avgLBTs.append(avgLBT)

            stdLBT = np.std(LBT)
            seLBT = stdLBT / float(numFolds)

            seLBTs.append(seLBT)

            avgLBS = average(LBS)
            avgLBSs.append(avgLBS)

            stdLBS = np.std(LBS)
            seLBS = stdLBS / float(numFolds)

            seLBSs.append(seLBS)

            avgLRF = average(LRF)
            avgLRFs.append(avgLRF)

            stdLRF = np.std(LRF)
            seLRF = stdLRF / float(numFolds)

            seLRFs.append(seLRF)

        print "=======SVM======"
        print avgLSVMs
        print seLSVMs
        print "=======DT======"
        print avgLDTs
        print seLDTs
        print "=======BT======"
        print avgLBTs
        print seLBTs
        print "=======RF======"
        print avgLRFs
        print seLRFs
        print "=======BS======"
        print avgLBSs
        print seLBSs

        plt.figure()
        plt.errorbar(DSS, avgLSVMs, seLSVMs, label='SVM')
        plt.errorbar(DSS, avgLDTs, seLDTs, label='DT')
        plt.errorbar(DSS, avgLBTs, seLBTs, label='BT')
        plt.errorbar(DSS, avgLBSs, seLBSs, label='BS')
        plt.errorbar(DSS, avgLRFs, seLRFs, label='RF')
        plt.ylabel('Zero-one Loss')
        plt.xlabel('Depth limits')
        plt.title('Zero One loss for different depth limits ')
        plt.legend()
        axes = plt.gca()
        axes.set_ylim([0,0.6])
        plt.savefig('Analysis3.png')
        plt.show()
        plt.close()

    if ExploreTreeSpace:

        trainingFileName = arg[1]
        csvContent = readCSVFile(trainingFileName)
        randIDLists = getRandomIDList(len(csvContent))

        for countRevs in range(0,len(csvContent)):
            csvContent[countRevs][2] = csvContent[countRevs][2].lower().translate(None, string.punctuation).split()

        avgLSVMs = []
        seLSVMs = []
        avgLDTs = []
        seLDTs = []
        avgLBTs = []
        seLBTs = []
        avgLBSs = []
        seLBSs = []
        avgLRFs = []
        seLRFs = []
        NTS = []
        for counterTrees in range(0,len(NumTreesList)):
            print "======================================="
            LDT = []
            LSVM = []
            LBT = []
            LBS = []
            LRF = []
            for fold in range(0,numFolds):

                testSetIDs = randIDLists[fold]
                restLists = randIDLists[0:fold]+randIDLists[fold+1:]
                restIDs = [item for sublist in restLists for item in sublist]

                numTrainingEx = int((percentTraining[3] / 100.0) * len(csvContent))
                randIdsTr = random.sample(restIDs, numTrainingEx)

                if fold == 0:
                    NTS.append(NumTreesList[counterTrees])

                trainingContent = []
                testingContent = []

                for i in range(0,len(randIdsTr)):
                    temp = csvContent[randIdsTr[i]]
                    trainingContent.append(temp)
                for i in range(0,len(testSetIDs)):
                    temp = csvContent[testSetIDs[i]]
                    testingContent.append(temp)

                uniqueWordsFreqsTuple = getUniqueWords(trainingContent)
                uniqueWordsFreqsTuple = uniqueWordsFreqsTuple\
                    [numWordsToDiscardTop:numWordsToDiscardTop+featureSizeList[2]]
                listWords = [i for (i,j) in uniqueWordsFreqsTuple]

                featuresTraining = constructFeatures(trainingContent, listWords)
                featuresTraining = featuresTraining.astype(float)

                featuresTesting = constructFeatures(testingContent, listWords)
                featuresTesting = featuresTesting.astype(float)

                featuresTestingDT = prepareDataDT(testingContent, listWords)

                classLabelsTr = getClassLabels(trainingContent)
                classLabelsTr = classLabelsTr.astype(float)

                classLabelsTs = getClassLabels(testingContent)
                classLabelsTs = classLabelsTs.astype(float)

                rootNode = splitData(featuresTraining, classLabelsTr, 0)
                buildTree(rootNode, featuresTraining, classLabelsTr, 1, 0,DTDEPTHLIMIT)

                predictLabelsDT = predictUsingDT(featuresTestingDT, rootNode)

                zeroOneLossDT = findZeroOneLoss(predictLabelsDT,classLabelsTs)
                print 'ZERO-ONE-LOSS-DT ' + str(zeroOneLossDT)
                LDT.append(zeroOneLossDT)

                roots = []
                for countTree in range(0,NumTreesList[counterTrees]):
                    randIdsEgs = np.random.choice(range(0,len(trainingContent)),len(trainingContent),replace=True)
                    temp = []
                    for i in range(0,len(randIdsEgs)):
                        temp.append(trainingContent[randIdsEgs[i]])

                    featuresTraining = constructFeatures(temp, listWords)
                    classLabelsTr = getClassLabels(temp)

                    rootNode = splitData(featuresTraining, classLabelsTr, 0)
                    buildTree(rootNode, featuresTraining, classLabelsTr, 1, 0,DTDEPTHLIMIT)

                    roots.append(rootNode)

                featuresTesting = prepareDataDT(testingContent,listWords)
                classLabelsTs = getClassLabels(testingContent)
                classLabelsTs = classLabelsTs.astype(float)

                predsAllTrees = []
                for j in range(0,NumTreesList[counterTrees]):
                    predictLabelsDT = predictUsingDT(featuresTesting, roots[j])
                    predsAllTrees.append(predictLabelsDT)
                finalPreds = finalPredsBT(predsAllTrees)
                zeroOneLossBT = findZeroOneLoss(finalPreds,classLabelsTs)
                print 'ZERO-ONE-LOSS-BT ' + str(zeroOneLossBT)
                LBT.append(zeroOneLossBT)

                roots = []
                for countTree in range(0,NumTreesList[counterTrees]):
                    randIdsEgs = np.random.choice(range(0,len(trainingContent)),len(trainingContent),replace=True)
                    temp = []
                    for i in range(0,len(randIdsEgs)):
                        temp.append(trainingContent[randIdsEgs[i]])

                    featuresTraining = constructFeatures(temp, listWords)
                    classLabelsTr = getClassLabels(temp)

                    rootNode = splitData(featuresTraining, classLabelsTr, 1)
                    buildTree(rootNode, featuresTraining, classLabelsTr, 1, 1,DTDEPTHLIMIT)

                    roots.append(rootNode)

                featuresTesting = prepareDataDT(testingContent,listWords)
                classLabelsTs = getClassLabels(testingContent)
                classLabelsTs = classLabelsTs.astype(float)

                predsAllTrees = []
                for j in range(0,NumTreesList[counterTrees]):
                    predictLabelsDT = predictUsingDT(featuresTesting, roots[j])
                    predsAllTrees.append(predictLabelsDT)
                finalPreds = finalPredsBT(predsAllTrees)
                zeroOneLossRF = findZeroOneLoss(finalPreds,classLabelsTs)
                print 'ZERO-ONE-LOSS-RF ' + str(zeroOneLossRF)
                LRF.append(zeroOneLossRF)

                roots = []
                am = []
                ws = np.ones((len(trainingContent)),dtype=float) / float(len(trainingContent))
                featuresTrainingAlldata = prepareDataDT(trainingContent, listWords)
                classLabelsTrAlldata = getClassLabels(trainingContent)

                allEgs = np.arange(0,len(trainingContent))
                for countTree in range(0,NumTreesList[counterTrees]):
                    randomEgs = np.random.choice(allEgs,len(trainingContent), replace=True, p = ws)

                    trainData = []
                    for cEg in range(0, len(randomEgs)):
                        trainData.append(trainingContent[randomEgs[cEg]])

                    featuresTraining = constructFeatures(trainData, listWords)
                    classLabelsTr = getClassLabels(trainData)

                    rootNode = splitData(featuresTraining, classLabelsTr, 0)
                    buildTree(rootNode, featuresTraining, classLabelsTr, 1, 0,DTDEPTHLIMIT)
                    roots.append(rootNode)

                    classLabelsTrPlusMinus = classLabelsTrAlldata
                    classLabelsTrPlusMinus[classLabelsTrPlusMinus==0.0] = -1.0

                    predictLabelsDT = predictUsingDT(featuresTrainingAlldata, rootNode)
                    predictLabelsDTPlusMinus = predictLabelsDT
                    predictLabelsDTPlusMinus[predictLabelsDTPlusMinus==0.0] = -1.0

                    err = findError(predictLabelsDTPlusMinus, classLabelsTrPlusMinus, ws)
                    alpha = 0.5 * np.log((1-err)/err)
                    am.append(alpha)

                    ws = updateWeightsBS(ws, alpha, predictLabelsDTPlusMinus, classLabelsTrPlusMinus)

                featuresTesting = prepareDataDT(testingContent,listWords)
                classLabelsTs = getClassLabels(testingContent)
                classLabelsTs = classLabelsTs.astype(float)

                predsAllTrees = []
                for j in range(0,NumTreesList[counterTrees]):
                    predictLabelsDT = predictUsingDT(featuresTesting, roots[j])
                    predictLabelsDT[predictLabelsDT==0.0] = -1.0
                    predsAllTrees.append(predictLabelsDT)
                finalPreds = finalPredsBS(predsAllTrees,am)
                zeroOneLossBS = findZeroOneLoss(finalPreds,classLabelsTs)
                print 'ZERO-ONE-LOSS-BS ' + str(zeroOneLossBS)
                LBS.append(zeroOneLossBS)

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
                LSVM.append(zeroOneLossSVM)

            avgLSVM = average(LSVM)
            avgLSVMs.append(avgLSVM)

            stdLSVM = np.std(LSVM)
            seLSVM = stdLSVM / float(numFolds)

            seLSVMs.append(seLSVM)

            avgLDT = average(LDT)
            avgLDTs.append(avgLDT)

            stdLDT = np.std(LDT)
            seLDT = stdLDT / float(numFolds)

            seLDTs.append(seLDT)

            avgLBT = average(LBT)
            avgLBTs.append(avgLBT)

            stdLBT = np.std(LBT)
            seLBT = stdLBT / float(numFolds)

            seLBTs.append(seLBT)

            avgLBS = average(LBS)
            avgLBSs.append(avgLBS)

            stdLBS = np.std(LBS)
            seLBS = stdLBS / float(numFolds)

            seLBSs.append(seLBS)

            avgLRF = average(LRF)
            avgLRFs.append(avgLRF)

            stdLRF = np.std(LRF)
            seLRF = stdLRF / float(numFolds)

            seLRFs.append(seLRF)

        print "=======SVM======"
        print avgLSVMs
        print seLSVMs
        print "=======DT======"
        print avgLDTs
        print seLDTs
        print "=======BT======"
        print avgLBTs
        print seLBTs
        print "=======RF======"
        print avgLRFs
        print seLRFs
        print "=======BS======"
        print avgLBSs
        print seLBSs

        plt.figure()
        plt.errorbar(NTS, avgLSVMs, seLSVMs, label='SVM')
        plt.errorbar(NTS, avgLDTs, seLDTs, label='DT')
        plt.errorbar(NTS, avgLBTs, seLBTs, label='BT')
        plt.errorbar(NTS, avgLBSs, seLBSs, label='BS')
        plt.errorbar(NTS, avgLRFs, seLRFs, label='RF')
        plt.ylabel('Zero-one Loss')
        plt.xlabel('Number of trees')
        plt.title('Zero One loss for different number of trees ')
        plt.legend()
        axes = plt.gca()
        axes.set_ylim([0,0.6])
        plt.savefig('Analysis4.png')
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
        uniqueWordsFreqsTuple = uniqueWordsFreqsTuple\
            [numWordsToDiscardTop:numWordsToDiscardTop+numWordsToConstructFeat]
        listWords = [i for (i,j) in uniqueWordsFreqsTuple]

        #Afterwards we construct features from the training and testing
        #content. Specifically, these features set is a matrix of
        #size number of words considered times number of reviews in content.

        if int(modelChoice) == 1: #DT
            featuresTraining = constructFeatures(trainingContent,listWords)
            classLabelsTr = getClassLabels(trainingContent)

            rootNode = splitData(featuresTraining, classLabelsTr, 0)
            buildTree(rootNode, featuresTraining, classLabelsTr, 1, 0,DTDEPTHLIMIT)

            featuresTesting = prepareDataDT(testingContent,listWords)
            classLabelsTs = getClassLabels(testingContent)
            classLabelsTs = classLabelsTs.astype(float)

            predictLabelsDT = predictUsingDT(featuresTesting, rootNode)
            # print predictLabelsDT
            # print classLabelsTs
            zeroOneLossDT = findZeroOneLoss(predictLabelsDT,classLabelsTs)
            print 'ZERO-ONE-LOSS-DT ' + str(zeroOneLossDT)

        if int(modelChoice) == 2: #BT
            roots = []
            for countTree in range(0,BTNUMTREES):
                print  "tree number: " + str(countTree)
                randIdsEgs = np.random.choice(range(0,len(trainingContent)),len(trainingContent),replace=True)
                temp = []
                for i in range(0,len(randIdsEgs)):
                    temp.append(trainingContent[randIdsEgs[i]])

                featuresTraining = constructFeatures(temp, listWords)
                classLabelsTr = getClassLabels(temp)

                rootNode = splitData(featuresTraining, classLabelsTr, 0)
                buildTree(rootNode, featuresTraining, classLabelsTr, 1, 0,DTDEPTHLIMIT)

                roots.append(rootNode)

            featuresTesting = prepareDataDT(testingContent,listWords)
            classLabelsTs = getClassLabels(testingContent)
            classLabelsTs = classLabelsTs.astype(float)

            predsAllTrees = []
            for j in range(0,BTNUMTREES):
                predictLabelsDT = predictUsingDT(featuresTesting, roots[j])
                predsAllTrees.append(predictLabelsDT)
            finalPreds = finalPredsBT(predsAllTrees)
            zeroOneLossBT = findZeroOneLoss(finalPreds,classLabelsTs)
            print 'ZERO-ONE-LOSS-BT ' + str(zeroOneLossBT)

        if int(modelChoice) == 3: #RF
            roots = []
            for countTree in range(0,RFNUMTREES):
                print  "tree number: " + str(countTree)
                randIdsEgs = np.random.choice(range(0,len(trainingContent)),len(trainingContent),replace=True)
                temp = []
                for i in range(0,len(randIdsEgs)):
                    temp.append(trainingContent[randIdsEgs[i]])

                featuresTraining = constructFeatures(temp, listWords)
                classLabelsTr = getClassLabels(temp)

                rootNode = splitData(featuresTraining, classLabelsTr, 1)
                buildTree(rootNode, featuresTraining, classLabelsTr, 1, 1,DTDEPTHLIMIT)

                roots.append(rootNode)

            featuresTesting = prepareDataDT(testingContent,listWords)
            classLabelsTs = getClassLabels(testingContent)
            classLabelsTs = classLabelsTs.astype(float)

            predsAllTrees = []
            for j in range(0,RFNUMTREES):
                predictLabelsDT = predictUsingDT(featuresTesting, roots[j])
                predsAllTrees.append(predictLabelsDT)
            finalPreds = finalPredsBT(predsAllTrees)
            zeroOneLossRF = findZeroOneLoss(finalPreds,classLabelsTs)
            print 'ZERO-ONE-LOSS-RF ' + str(zeroOneLossRF)

        if int(modelChoice) == 4: #BS
            roots = []
            am = []
            ws = np.ones((len(trainingContent)),dtype=float) / float(len(trainingContent))
            featuresTrainingAlldata = prepareDataDT(trainingContent, listWords)
            classLabelsTrAlldata = getClassLabels(trainingContent)

            allEgs = np.arange(0,len(trainingContent))
            for countTree in range(0,BSNUMTREES):
                print "tree number: " + str(countTree)
                randomEgs = np.random.choice(allEgs,len(trainingContent), replace=True, p = ws)

                trainData = []
                for cEg in range(0, len(randomEgs)):
                    trainData.append(trainingContent[randomEgs[cEg]])

                featuresTraining = constructFeatures(trainData, listWords)
                classLabelsTr = getClassLabels(trainData)

                rootNode = splitData(featuresTraining, classLabelsTr, 0)
                buildTree(rootNode, featuresTraining, classLabelsTr, 1, 0,DTDEPTHLIMIT)
                roots.append(rootNode)

                classLabelsTrPlusMinus = classLabelsTrAlldata
                classLabelsTrPlusMinus[classLabelsTrPlusMinus==0.0] = -1.0

                predictLabelsDT = predictUsingDT(featuresTrainingAlldata, rootNode)
                predictLabelsDTPlusMinus = predictLabelsDT
                predictLabelsDTPlusMinus[predictLabelsDTPlusMinus==0.0] = -1.0

                err = findError(predictLabelsDTPlusMinus, classLabelsTrPlusMinus, ws)
                alpha = 0.5 * np.log((1-err)/err)
                am.append(alpha)

                ws = updateWeightsBS(ws, alpha, predictLabelsDTPlusMinus, classLabelsTrPlusMinus)

            featuresTesting = prepareDataDT(testingContent,listWords)
            classLabelsTs = getClassLabels(testingContent)
            classLabelsTs = classLabelsTs.astype(float)

            predsAllTrees = []
            for j in range(0,BSNUMTREES):
                predictLabelsDT = predictUsingDT(featuresTesting, roots[j])
                predictLabelsDT[predictLabelsDT==0.0] = -1.0
                predsAllTrees.append(predictLabelsDT)
            finalPreds = finalPredsBS(predsAllTrees,am)
            zeroOneLossBS = findZeroOneLoss(finalPreds,classLabelsTs)
            print 'ZERO-ONE-LOSS-BS ' + str(zeroOneLossBS)

        if int(modelChoice) == 5: #SVM
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
            print 'training svm'
            SVMWeights = getSVMWeights(featuresTraining, classLabelsTr)
            predLabelsSVM = predictUsingSVM(featuresTesting,SVMWeights)
            predLabelsSVM = np.array(predLabelsSVM)
            classLabelsTs[classLabelsTs==0.0] = -1.0

            zeroOneLossSVM = findZeroOneLoss(predLabelsSVM,classLabelsTs)
            print 'ZERO-ONE-LOSS-SVM ' + str(zeroOneLossSVM)

if __name__ == "__main__":
    main(sys.argv[0:])
