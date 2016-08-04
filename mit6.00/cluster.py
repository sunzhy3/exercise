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
        
class Mammal(Point):
    def __init__(self, name, originalAttrs, scaleAttrs = None):
        Point.__init__(self, name, originalAttrs, originalAttrs)
    def scaleFeatures(self, key):
        scaleDict = {'identity':[1, 1, 1, 1, 1, 1, 1, 1],
                      '1/max':[1/3.0, 1/4.0, 1.0, 1/4.0, 1/4.0, 1/6.0, 1/6.0]}
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
def test0(numClusters = 2, scaling = 'identity', printSteps = False, printHistory = True):
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

test0()

def kmeans(points, k, cutoff, pointType, maxIters = 100, toPoint = False):
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
        newCluster = []
        for i in xrange(k):
            newClusters.append([])
        for p in points:
            #Find the centroid closest to p
            smallestDistance = p.distance(clusters[0].getCentroid())
            index = 0
            for i in xrange(k):
                distance = p.distance(clusters[0].getCentroid())
                if distance < smallestDistance:
                    smallestDistance = distance
                    index = i
            #Add p to the list of points for the appropriate cluster
            newClusters[index].append(p)
        #update each cluster and record how much the centeoid has changed
        biggestChange = 0.0
        for i in xrange(len(cluster)):
            change = cluster[i].update(newClusters[i])
            biggestChange = max(biggestChange, change)
        numIters += 1
    #calculate the coherence of the least coherent cluster
    maxDist = 0.0
    for c in clusters:
        for p in c.members():
            if p.distance(c.getCentroid()) >maxDist:
                maxDist = p.diatance(c.getCentroid())
    print 'Number of iterations = ', numIters, 'Max Diameter =', maxDist
    return clusters, maxDist

#US counties example
class County(Point):
    #Interesting subsets of features
    #0 = don't use, 1 = use
    noWealth = (('HomeVal', '0'), ('Income', '0'), ('Population', '1'),
                ('Pop Change', '1'), ('Prcnt 65+', '1'), ('Below 18', '1'),
                ('Prcnt Female', '1'), ('Prcnt HS Grad', '1'),
                ('Prcnt College', '1'), ('Unemployed', '0'),
                ('Prcnt Below 18', '1'), ('Life Expect', '1'),
                ('Farm Acres', '1'))
    wealthOnly = (('HomeVal', '1'), ('Income', '1'), ('Population', '1'),
                ('Pop Change', '0'), ('Prcnt 65+', '0'), ('Below 18', '0'),
                ('Prcnt Female', '0'), ('Prcnt HS Grad', '0'),
                ('Prcnt College', '0'), ('Unemployed', '1'),
                ('Prcnt Below 18', '0'), ('Life Expect', '0'),
                ('Farm Acres', '0'))
    education =  (('HomeVal', '0'), ('Income', '0'), ('Population', '0'),
                ('Pop Change', '0'), ('Prcnt 65+', '0'), ('Below 18', '0'),
                ('Prcnt Female', '0'), ('Prcnt HS Grad', '1'),
                ('Prcnt College', '1'), ('Unemployed', '0'),
                ('Prcnt Below 18', '0'), ('Life Expect', '0'),
                ('Farm Acres', '0'))
    allFeatures = (('HomeVal', '1'), ('Income', '1'), ('Population', '1'),
                ('Pop Change', '1'), ('Prcnt 65+', '1'), ('Below 18', '1'),
                ('Prcnt Female', '1'), ('Prcnt HS Grad', '1'),
                ('Prcnt College', '1'), ('Unemployed', '1'),
                ('Prcnt Below 18', '1'), ('Life Expect', '1'),
                ('Farm Acres', '1'))
    filterNames = {'all':allfeatures, 'education':education,
                  'wealthOnly':wealthOnly, 'noWealth':noWealth}
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
            for i in xrange(len(County.attrsFilter)):
                if Conty.attrsFilter[i] == 1
                    features.append(self)
    









    
