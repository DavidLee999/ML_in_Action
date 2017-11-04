from numpy import *

def loadDataSet():
    return [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]

def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1: # create sets with only one elements
                C1.append([item])

    C1.sort()
    return map(frozenset, C1)

def scanD(D, Ck, minSupport):
    ssCnt = {}
    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if not ssCnt.has_key(can):
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1

    numItems = float(len(D))
    retList = []
    supportData = {}
    for key in ssCnt:
        support = ssCnt[key] / numItems
        if support >= minSupport:
            retList.insert(0, key)
        supportData[key] = support

    return retList, supportData

def aprioriGen(Lk, k):
    retList = []
    lenLk = len(Lk)

    for i in range(lenLk):
        for j in range(i + 1, lenLk): # combine two sets with the same first k elements
            L1 = list(Lk[i])[:k - 2] # why it works: Apriori principle: if an itemset is infrequent, then its
            L2 = list(Lk[j])[:k - 2] # supersets are also infrequent. And we generate sets with more items by
            L1.sort() # combining sets with less items, if one of the subset do not exist, meaning its supperset
            L2.sort() # is not a frequent set

            if L1 == L2:
                retList.append(Lk[i] | Lk[j])

    return retList

def apriori(dataSet, minSupport = 0.5):
    C1 = createC1(dataSet)
    D = map(set, dataSet)
    L1, supportData = scanD(D, C1, minSupport)
    L = [L1]
    k = 2
    while (len(L[k - 2]) > 0):
        Ck = aprioriGen(L[k - 2], k)
        print Ck
        Lk, supK = scanD(D, Ck, minSupport)
        supportData.update(supK)
        L.append(Lk)
        k += 1

    return L, supportData

def generateRules(L, supportData, minConf = 0.7):
    bigRuleList = []
    for i in range(1, len(L)): # start from more-than-one-itme sets
        for freqSet in L[i]:
            H1 = [frozenset([item]) for item in freqSet]

            if (i > 1):
                rulesFromConseq(freqSet, H1, supportData, bigRuleList, minConf)
            else: # freq with more than 2 items
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)

    return bigRuleList

def calcConf(freqSet, H, supportData, brl, minConf = 0.7):
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq] # calculating support(p|h) / support(h)
        if conf > minConf:
            print freqSet - conseq, '-->', conseq, 'conf: ', conf
            brl.append((freqSet - conseq, conseq, conf))
            prunedH.append(conseq)

    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf = 0.7):
    m = len(H[0])
    if (len(freqSet) > (m + 1)): # if freqSet is large enough to remove H
        Hmp1 = aprioriGen(H, m + 1) # generate superset of H
        Hmp1 = calcConf(freqSet, Hmp1, supportData, brl, minConf) # confidence of h with more than 1 item
        if (len(Hmp1) > 1):
            rulesFromConseq(freqSet, Hmp1, supportData, brl, minConf) # recursive until h with only one item
