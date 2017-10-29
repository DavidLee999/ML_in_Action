from numpy import *

def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t')) - 1
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')

        for i in range(numFeat):
            lineArr.append(float(curLine[i]))

        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))

    return dataMat, labelMat

def standRegres(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T

    xTx = xMat.T * xMat
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse."
        return

    ws = xTx.I * (xMat.T * yMat)

    return ws

def lwlr(testPoint, xArr, yArr, k = 1.0):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    m = shape(xMat)[0]
    weights = mat(eye((m)))

    for j in range(m):
        diffMat = testPoint - xMat[j, :]
        weights[j, j] = exp(diffMat * diffMat.T / (-2.0 * k ** 2))

    xTx = xMat.T * (weights * xMat)
    if linalg.det(xTx) == 0.0:
        print "This matrix is singular, cannot do inverse!"
        return

    ws = xTx.I * (xMat.T * (weights * yMat))

    return testPoint * ws

def lwlrTest(testArr, xArr, yArr, k = 1.0):
    m = shape(testArr)[0]
    yHat = zeros(m)
    
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)

    return yHat

def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()

def ridgeRegres(xMat, yMat, lam = 0.2):
    xTx = xMat.T * xMat
    denom = xTx + eye(shape(xMat)[1]) * lam
    if linalg.det(denom) == 0.0:
        print "this matrix is singular, cannot do inverse"
        return

    ws = denom.I * (xMat.T * yMat)

    return ws

def ridgeTest(xArr, yArr):
    xMat = mat(xArr)
    yMat = mat(yArr).T
    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMeans = mean(xMat, 0)
    xVar = var(xMat, 0)
    xMat = (xMat - xMeans) / xVar
    numTestPts = 30
    wMat = zeros((numTestPts, shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat, yMat, exp(i - 10))
        wMat[i, :] = ws.T

    return wMat

def regularize(xMat):
    inMat = xMat.copy()
    inMeans = mean(inMat, 0)
    inVar = var(inMat, 0)
    inMat = (inMat - inMeans) / inVar

    return inMat

def stageWise(xArr, yArr, eps = 0.01, numIt = 100):
    xMat = mat(xArr)
    yMat = mat(yArr).T

    yMean = mean(yMat, 0)
    yMat = yMat - yMean
    xMat = regularize(xMat)

    m, n = shape(xMat)
    returnMat = zeros((numIt, n))
    ws = zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()

    for i in range(numIt):
        # print ws.T
        lowestError = inf
        for j in range(n): # greedy algorithm
            for sign in [-1, 1]: # increase or decrease the weights
                wsTest = ws.copy()
                wsTest[j] += eps * sign # increase or decrease only one weight
                yTest = xMat * wsTest # test the performance
                rssE = rssError(yMat.A, yTest.A)
                if rssE < lowestError: # if better
                    lowestError = rssE
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T

    return returnMat

from bs4 import BeautifulSoup

def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    fr = open(inFile)
    soup = BeautifulSoup(fr.read())
    i = 1
    
    currentRow = soup.findAll('table', r = '%d' %i)
    while (len(currentRow) != 0):
        currentRpw = soup.findAll('table', r = '%d' %i)
        title = currentRow[0].findAll('a')[1].text
        lwrTitle = title.lower()
        
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
           
        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
        if len(soldUnicde) == 0:
            print "item #%d did not sell" %i
        else:
            soldPrice = currentRow[0].findAll('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$','')
            priceStr = priceStr.replace(',','')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            if  sellingPrice > origPrc * 0.5:
                print "%d\t%d\t%d\t%f\t%f" % (yr,numPce,newFlag,origPrc, sellingPrice)
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        
        currentRow = soup.findAll('table', r="%d" % i)

def setDataCollect(retX, retY):
    scrapePage(retX, retY, 'setHtml/lego8288.html', 2006, 800, 49.99)
    scrapePage(retX, retY, 'setHtml/lego10030.html', 2002, 3096, 269.99)
    scrapePage(retX, retY, 'setHtml/lego10179.html', 2007, 5195, 499.99)
    scrapePage(retX, retY, 'setHtml/lego10181.html', 2007, 3428, 199.99)
    scrapePage(retX, retY, 'setHtml/lego10189.html', 2008, 5922, 299.99)
    scrapePage(retX, retY, 'setHtml/lego10196.html', 2009, 3263, 249.99)
    
def crossValidation(xArr, yArr, numVal = 10):
    m = len(yArr)
    indexList = range(m)
    errorMat = zeros((numVal, 30))
    
    for i in range(numVal):
        trainX = []
        trainY = []
        testX = []
        testY = []
        
        random.shuffle(indexList)
        for j in range(m): # 10-folder cross validation
            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        
        wMat = ridgeTest(trainX, trainY)
        for k in range(30):
            matTestX = mat(testX)
            matTrainX = mat(trainX)
            meanTrain = mean(matTrainX, 0)
            varTrain = var(matTrainX, 0)
            
            matTestX = (matTestX - meanTrain) / varTrain # regularize test data as the training data does
            yEst = matTestX * mat(wMat[k, :]).T + mean(trainY)
            errorMat[i, k] = rssError(yEst.T.A, array(testY))
    
    meanErrors = mean(errorMat, 0)
    minMean = float(min(meanErrors))
    bestWeights = wMat[nonzero(meanErrors == minMean)]
    
    xMat = mat(xArr)
    yMat = mat(yArr).T
    meanX = mean(xMat, 0)
    varX = var(xMat, 0)
    unReg = bestWeights / varX
    
    print "the best model from Ridge Regression is \n", unReg
    print "with constant term: ", -1 * sum(multiply(meanX, unReg)) + mean(yMat)