class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLike = None
        self.parent = parentNode
        self.children = {}

    def inc(self, numOccur):
        self.count += numOccur

    def disp(self, ind = 1):
        print '  ' * ind, self.name, ' ', self.count
        for child in self.children.values():
            child.disp(ind + 1)

def createTree(dataSet, minSup = 1):
    headerTable = {}
    for trans in dataSet:
        for item in trans:
            headerTable[item] = headerTable.get(item, 0) + dataSet[trans]

    for k in headerTable.keys(): # find frequent items
        if headerTable[k] < minSup:
            del(headerTable[k])

    freqItemSet = set(headerTable.keys())
    if len(freqItemSet) == 0:
        return None, None

    for k in headerTable: # building head table
        headerTable[k] = [headerTable[k], None]
    
    retTree = treeNode('Null Set', 1, None)
    for tranSet, count in dataSet.items():
        localID = {}
        for item in tranSet: # filter frequent itemset
            if item in freqItemSet:
                localID[item] = headerTable[item][0]

        if len(localID) > 0: # sort
            orderedItems = [v[0] for v in sorted(localID.items(), key = lambda p: p[1], reverse = True)]
            updateTree(orderedItems, retTree, headerTable, count) # add them to the tree

    return retTree, headerTable

def updateTree(items, inTree, headerTable, count):
    if items[0] in inTree.children: # the first item is already one of the children of the parent node
        inTree.children[items[0]].inc(count)
    else:
        inTree.children[items[0]] = treeNode(items[0], count, inTree) # if not, build a new node
        if headerTable[items[0]][1] == None: # add the first existing item to the head table
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:
            updateHeader(headerTable[items[0]][1], inTree.children[items[0]]) # update the list

    if len(items) > 1: # recursively add the tree node
        updateTree(items[1::], inTree.children[items[0]], headerTable, count)

def updateHeader(nodeToTest, targetNode):
    while (nodeToTest.nodeLike != None):
        nodeToTest = nodeToTest.nodeLike
    nodeToTest.nodeLike = targetNode

def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1

    return retDict

def ascendTree(leafNode, prefixPath):
    if leafNode.parent != None: # recrusively find the parent node until to the root
        prefixPath.append(leafNode.name)
        ascendTree(leafNode.parent, prefixPath)

def findPrefixPath(basePat, treeNode):
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTree(treeNode, prefixPath)

        if len(prefixPath) > 1: # if the tree node is not the child of the root
            condPats[frozenset(prefixPath[1:])] = treeNode.count # build the cond. pattern, exclused itself
        treeNode = treeNode.nodeLike # process the node with the same name

    return condPats

def mineTree(inTree, headerTable, minSup, preFix, freqItemList):
    bigL = [v[0] for v in sorted(headerTable.items(), key = lambda p: p[1])] # reverse the head table

    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat, headerTable[basePat][1])
        myCondTree, myHead = createTree(condPattBases, minSup) # create conditional FP tree

        if myHead != None: # recursively find frequent itemsets unitl there is no node in the c-FP tree
            mineTree(myCondTree, myHead, minSup, newFreqSet, freqItemList)

