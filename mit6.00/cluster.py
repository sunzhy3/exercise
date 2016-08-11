import pylab, random, string, copy

class Point(object):
    def __init__(self, name, originalAttrs, normalizedAttrs = None):
        """normalizedAttrs and orignalAttrs are both arrays"""
        self.name = name
        self.unNormalized = originalAttrs
        if normalizedAttrs == None:
            self.attrs = originalAttrs
        else:
            self.attrs = normalizedAttrs
    def dimensionality(self):
        return len(self.attrs)
    def getAttrs(self):
        return self.attrs
    def getOriginalAttrs(self):
        return self.unNormalized
    def distance(self, other):
        #Euclidean distance metric
        result = 0.0
        for i in xrange(self.dimensionality()):
            result += (self.attrs[i] - other.attrs[i]) ** 2
        return result ** 0.5
    def getName(self):
        return self.name
    def __str__(self):
        return self.name

class Cluster(object):
    def __init__(self, points, pointType):
        self.points = points
        self.pointType = pointType
        self.centroid = self.computeCentroid()
    def singleLinkageDist(self, other):
        minDist = self.points[0].distance(other.points[0])
        for p1 in self.points:
            for p2 in other.points:
                if p1.distance(p2) < minDist:
                    minDist = p1.distance(p2)
        return minDist
    def maxLinkageDist(self, other):
        maxDist = self.points[0].distance(other.points[0])
        for p1 in self.points:
            for p2 in other.points:
                if p1.distance(p2) < maxDist:
                    maxDist = p1.distance(p2)
        return maxDist
    def averageLinkageDist(self, other):
        totDist = 0.0
        for p1 in self.points:
            for p2 in other.points:
                totDist += p1.distance(p2)
        return totDist / (len(self.points) * len(other.points))
    def update(self, points):
        oldCentroid = self.centroid
        self.points = points
        if len(points) > 0:
            self.centroid = self.computeCentroid()
            return oldCentroid.distance(self.centroid)
        else:
            return 0.0
    def members(self):
        for p in self.points:
            yield p
    def isIn(self, name):
        for p in self.points:
            if p.getName() == name:
                return True
        return False
    def toStr(self):
        result = ''
        for p in self.points:
            result = result + p.toStr() + ','#something weird here
        return result[: -2]
    def __str__(self):
        names = []
        for p in self.points:
            names.append(p.getName())
        names.sort()
        result = ''
        for p in names:
            result = result + p + ', '
        return result[:-2]
    def getCentroid(self):
        return self.centroid
    def computeCentroid(self):
        dim = self.points[0].dimensionality()
        totVals = pylab.array([0.0] * dim)
        for p in self.points:
            totVals += p.getAttrs()
        centroid = self.pointType('mean',
                                  totVals/float(len(self.points)),
                                  totVals/float(len(self.points)))
        return centroid

class ClusterSet(object):
    def __init__(self, pointType):
        self.members = []
    def add(self, c):
        if c in self.members:
            raise ValueError
        self.members.append(c)
    def getClusters(self):
        return self.members[:]
    def mergeClusters(self, c1, c2):
        points = []
        for p in c1.members():
            points.append(p)
        for p in c2.members():
            points.append(p)
        newC = Cluster(points, type(p))
        self.members.remove(c1)
        self.members.remove(c2)
        self.add(newC)
        return c1, c2
    def findClosest(self, metric):
        minDistance = metric(self.members[0], self.members[1])
        toMerge = (self.members[0], self.members[1])
        for c1 in self.members:
            for c2 in self.members:
                if c1 == c2:
                    continue
                if metric(c1, c2) <minDistance:
                    minDistance = metric(c1, c2)
                    toMerge = (c1, c2)
        return toMerge
    def mergeOne(self, metric, toPoint = False):
        if len(self.members) == 1:
            return None
        if len(self.members) == 2:
            return self.mergeClusters(self.members[0], self.members[1])
        toMerge = self.findClosest(metric)
        if toPoint:
            print 'Merged'
            print ' ' + str(toMerge[0])
            print 'with'
            print ' ' + str(toMerge[1])
        self.mergeClusters(toMerge[0], toMerge[1])
        return toMerge
    def mergeN(self, metric, numClusters = 1, history = [], toPrint = False):
        assert numClusters >= 1
        while len(self.members) > numClusters:
            merged = self.mergeOne(metric, toPrint)
            history.append(merged)
        return history
    def numClusters(self):
        return len(self.members) + 1
    def __str__(self):
        result = ''
        for c in self.members:
            result = result + str(c) + '\n'
        return result

#Mammal's teeth example
class Mammal(Point):
    def __init__(self, name, originalAttrs, scaleAttrs = None):
        Point.__init__(self, name, originalAttrs, originalAttrs)
    def scaleFeatures(self, key):
        scaleDict = {'identity':[1, 1, 1, 1, 1, 1, 1, 1],
                      '1/max':[1/3.0, 1/4.0, 1.0, 1.0,  1/4.0, 1/4.0, 1/6.0, 1/6.0]}
        scaleFeatures = []
        features = self.getOriginalAttrs()
        for i in xrange(len(features)):
            scaleFeatures.append(features[i] * scaleDict[key][i])
        self.attrs = scaleFeatures

def readMammalData(fName):
    dataFile = open(fName, 'r')
    teethList = []
    nameList = []
    for line in dataFile:
        if len(line) == 0 or line[0] =='#':
            continue
        dataline = string.split(line)
        teeth = dataline.pop(-1)
        features = []
        for t in teeth:
            features.append(float(t))
        name = ''
        for w in dataline:
            name = name + w + ' '
        name = name[:-1]
        teethList.append(features)
        nameList.append(name)
    return nameList, teethList

def buildMammalPoints(fName, scaling):
    nameList, featureList = readMammalData(fName)
    points = []
    for i in xrange(len(nameList)):
        point = Mammal(nameList[i], pylab.array(featureList[i]))
        point.scaleFeatures(scaling)
        points.append(point)
    return points

#Use hierarchical clustering for mammal teeth
def hiearchicalTest(numClusters = 2, scaling = 'identity',
                printSteps = False, printHistory = True):
    points = buildMammalPoints('/home/szy/exercise/mit6.00/mammalTeeth.txt', scaling)
    cS = ClusterSet(Mammal)
    for p in points:
        cS.add(Cluster([p], Mammal))
    history = cS.mergeN(Cluster.maxLinkageDist, numClusters, toPrint = printSteps)
    if printHistory:
        print ''
        for i in xrange(len(history)):
            names1 = []
            for p in history[i][0].members():
                names1.append(p.getName())
            names2 = []
            for p in history[i][1].members():
                names2.append(p.getName())
            print 'Step', i, 'Merged', names1, 'with', names2
            print ''
    clusters = cS.getClusters()
    print 'Final set of clusters:'
    index = 0
    for c in clusters:
        print ' C' + str(index) + ':', c
        index += 1

def kmeans(points, k, cutoff, pointType, maxIters = 100, toPrint = False):
    #Get k randomly chosen initial centroids
    initialCentroids = random.sample(points, k)
    clusters = []
    #Create a singleton cluster for each centroid
    for p in initialCentroids:
        clusters.append(Cluster([p], pointType))
    numIters = 0
    biggestChange = cutoff
    while biggestChange >= cutoff and numIters < maxIters:
        #Create a list contain k empty lists
        newClusters = []
        for i in xrange(k):
            newClusters.append([])
        for p in points:
            #Find the centroid closest to p
            smallestDistance = p.distance(clusters[0].getCentroid())
            index = 0
            for i in xrange(k):
                distance = p.distance(clusters[i].getCentroid())
                if distance < smallestDistance:
                    smallestDistance = distance
                    index = i
            #Add p to the list of points for the appropriate cluster
            newClusters[index].append(p)
        #update each cluster and record how much the centeoid has changed
        biggestChange = 0.0
        for i in xrange(len(clusters)):
            change = clusters[i].update(newClusters[i])
            biggestChange = max(biggestChange, change)
        numIters += 1
    #calculate the coherence of the least coherent cluster
    maxDist = 0.0
    for c in clusters:
        for p in c.members():
            if p.distance(c.getCentroid()) >maxDist:
                maxDist = p.distance(c.getCentroid())
    if toPrint:
        print 'Number of iterations = ', numIters, 'Max Diameter =', maxDist
    return clusters, maxDist

def kmeansTest(k = 2, cutoff = 0.0001, numTrials = 1,
               printSteps = False, printHistory = False):
    points = buildMammalPoints('/home/szy/exercise/mit6.00/mammalTeeth.txt', '1/max')
    if printSteps:
        print 'Points: '
        for p in points:
            attrs = p.getOriginalAttrs()
            for i in xrange(len(attrs)):
                attrs[i] = round(attrs[i], 2)
            print ' ', p, attrs
    numClusterings = 0
    bestDiameter = None
    while numClusterings <numTrials:
        clusters, maxDiameter = kmeans(points, k, cutoff, Mammal)
        if bestDiameter == None or maxDiameter < bestDiameter:
            bestDiameter = maxDiameter
            bestClustering = copy.deepcopy(clusters)
        if printHistory:
            print 'Cluster: '
            for i in range(len(clusters)):
                print ' C' + str(i) + ':' + cluster[i]
        numClusterings += 1
    print '\nBest Clustering'
    for i in xrange(len(bestClustering)):
        print ' C' + str(i) + ':', bestClustering[i]

#US counties example
class County(Point):
    #Interesting subsets of features
    #0 = don't use, 1 = use
    noWealth = (('HomeVal', '0'), ('Income', '0'),('Proverty', '0'),
                ('Population', '1'), ('Pop Change', '1'),
                ('Prcnt 65+', '1'), ('Below 18', '1'),
                ('Prcnt Female', '1'), ('Prcnt HS Grad', '1'),
                ('Prcnt College', '1'), ('Unemployed', '0'),
                ('Prcnt Below 18', '1'), ('Life Expect', '1'),
                ('Farm Acres', '1'))
    wealthOnly = (('HomeVal', '1'), ('Income', '1'), ('Proverty', '1'), 
                ('Population', '1'), ('Pop Change', '0'),
                ('Prcnt 65+', '0'), ('Below 18', '0'),
                ('Prcnt Female', '0'), ('Prcnt HS Grad', '0'),
                ('Prcnt College', '0'), ('Unemployed', '1'),
                ('Prcnt Below 18', '0'), ('Life Expect', '0'),
                ('Farm Acres', '0'))
    education =  (('HomeVal', '0'), ('Income', '0'),('Proverty', '0'),
                ('Population', '0'),('Pop Change', '0'),
                ('Prcnt 65+', '0'), ('Below 18', '0'),
                ('Prcnt Female', '0'), ('Prcnt HS Grad', '1'),
                ('Prcnt College', '1'), ('Unemployed', '0'),
                ('Prcnt Below 18', '0'), ('Life Expect', '0'),
                ('Farm Acres', '0'))
    allFeatures = (('HomeVal', '1'), ('Income', '1'), ('Proverty', '1'), 
                ('Population', '1'),('Pop Change', '1'),
                ('Prcnt 65+', '1'), ('Below 18', '1'),
                ('Prcnt Female', '1'), ('Prcnt HS Grad', '1'),
                ('Prcnt College', '1'), ('Unemployed', '1'),
                ('Prcnt Below 18', '1'), ('Life Expect', '1'),
                ('Farm Acres', '1'))
    noEducation = (('HomeVal', '1'), ('Income', '0'),('Proverty', '1'),
                ('Population', '1'),('Pop Change', '1'),
                ('Prcnt 65+', '1'), ('Below 18', '1'),
                ('Prcnt Female', '1'), ('Prcnt HS Grad', '0'),
                ('Prcnt College', '0'), ('Unemployed', '0'),
                ('Prcnt Below 18', '1'), ('Life Expect', '1'),
                ('Farm Acres', '1'))
    income = (('HomeVal', '0'), ('Income', '1'), ('Proverty', '1'), 
                ('Population', '0'),('Pop Change', '0'),
                ('Prcnt 65+', '0'), ('Below 18', '0'),
                ('Prcnt Female', '0'), ('Prcnt HS Grad', '0'),
                ('Prcnt College', '0'), ('Unemployed', '0'),
                ('Prcnt Below 18', '0'), ('Life Expect', '0'),
                ('Farm Acres', '0'))
    gender = (('HomeVal', '0'), ('Income', '0'), ('Proverty', '1'), 
                ('Population', '0'), ('Pop Change', '0'), 
                ('Prcnt 65+', '0'), ('Below 18', '0'),
                ('Prcnt Female', '1'), ('Prcnt HS Grad', '0'),
                ('Prcnt College', '1'), ('Unemployed', '0'),
                ('Prcnt Below 18', '0'), ('Life Expect', '0'),
                ('Farm Acres', '0'))
    filterNames = {'all':allFeatures, 'education':education,
                  'wealthOnly':wealthOnly, 'noWealth':noWealth,
                  'gender': gender, 'income': income,
                  'noEducation': noEducation}
    attrFilter = None
    #override Point to construct subset of features
    def __init__(self, name, originalAttrs, normalizedAttrs = None,
                 filterName = 'all'):
        if County.attrFilter == None:
            County.attrFilter = ''
            filterSpec = County.filterNames[filterName]
            for f in filterSpec:
                County.attrFilter += f[1]
        Point.__init__(self, name, originalAttrs, normalizedAttrs)
        features = []
        for i in xrange(len(County.attrFilter)):
            if County.attrFilter[i] == '1':
                features.append(self.getAttrs()[i])
        self.features = features
        #Override Point.distance to use only subset of features
        def distance(self, other):
            result = 0.0
            for i in xrange(len(self.features)):
                result += (self.features[i] - other.features[i]) ** 2
            return result ** 0.5

def readCountyData(fName, numEntries = 14):
    dataFile = open(fName, 'r')
    dataList = []
    nameList = []
    maxVals = pylab.array([0.0] * numEntries)
    #Build unnormalized feature vector
    for line in dataFile:
        if len(line) == 0 or line[0] == '#':
            continue
        dataLine = string.split(line)
        name = dataLine[0] + dataLine[1]
        features = []
        #Build features with numEntries features
        for f in dataLine[2:]:
            try:
                f = float(f)
                features.append(f)
                if f > maxVals[len(features) - 1]:
                    maxVals[len(features) - 1] = f
            except ValueError:
                name = name + f
        if len(features) != numEntries:
            continue
        dataList.append(features)
        nameList.append(name)
    dataFile.close()
    return nameList, dataList, maxVals

def buildCountyPoints(fName, filterName = 'all', scale = True):
    nameList, featureList, maxVals = readCountyData(fName)
    points = []
    for i in xrange(len(nameList)):
        originalAttrs = pylab.array(featureList[i])
        if scale:
            normalizedAttrs = originalAttrs / pylab.array(maxVals)
        else:
            normalizedAttrs = originalAttrs
        points.append(County(nameList[i], originalAttrs, normalizedAttrs, filterName))
    return points

def getAveIncome(cluster):
    tot = 0.0
    numElems = 0
    for c in cluster.members():
        tot += c.getOriginalAttrs()[1]
        numElems += 1
    if numElems > 0:
        return tot / numElems
    return 0.0

def featureKmeansTest(k = 50, cutoff = 0.01, numTrials = 1, myHome = 'MAMiddlesex',
                      filterName = 'all', printPoints = False, printClusters = False):
    #Build the set of points
    County.attrFilter = None
    points = buildCountyPoints('/home/szy/exercise/mit6.00/counties.txt', filterName)
    if printPoints:
        print 'Points'
        for p in points:
            attrs = p.getAttrs()
            for i in xrange(len(attrs)):
                attrs[i] = round(attrs[i], 2)
            print ' ', p, attrs
    print 'Cluster on', filterName
    numClusterings = 0
    bestDistance = None
    #Run k-means multiple times and choose best
    while numClusterings < numTrials:
        print 'Starting Clustering', numClusterings
        clusters, maxSmallest = kmeans(points, k, cutoff, County)
        numClusterings += 1
        if bestDistance == None or maxSmallest < bestDistance:
            bestDistance = maxSmallest
            bestClustering = copy.deepcopy(clusters)
        if printClusters:
            print 'Clusters:'
            for i in xrange(len(clusters)):
                print ' C' + str(i) + ':', clusters[i]
    for c in bestClustering:
        incomes = []
        for i in xrange(len(bestClustering)):
            incomes.append(getAveIncome(bestClustering[i]))
            if printClusters:
                print ' C' + str(i) + ':', clusters[i]
        pylab.hist(incomes)
        pylab.xlabel('Average Income')
        pylab.ylabel('Number of Clusters')
        pylab.title(filterName)
        if c.isIn(myHome):
            print 'Home Cluster:', c
            print 'Average Income of Home Cluster = ', round(getAveIncome(c), 0)

#kmeansTest(numTrials = 100)
#featureKmeansTest(k = 20, filterName = 'education', numTrials = 2)
featureKmeansTest(k = 20, filterName = 'gender', numTrials = 2)

pylab.show()








