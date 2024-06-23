import numpy as np
import sys
sys.path.insert(0, 'C:/Users/moyeradmin/Desktop/Coding Folder/Libraries/processpy')
import importlib
import processpy
importlib.reload(processpy)

import matplotlib.pyplot as plt

# The following classes are for orbit processing
class OrbitSegment:
    def __init__(self, startMarker, endMarker):
        self.startMarker = startMarker
        self.endMarker = endMarker
        self.startInd = startMarker.ind
        self.endInd = endMarker.ind
        self.segmentType = [0, 0]

        if self.startMarker.markerType == "Zero" and self.endMarker.markerType == "Min":
            self.segmentType = [0, -1]
        elif self.startMarker.markerType == "Zero" and self.endMarker.markerType == "Max":
            self.segmentType = [0, 1]
        elif self.startMarker.markerType == "Min" and self.endMarker.markerType == "Zero":
            self.segmentType = [-1, 0]
        elif self.startMarker.markerType == "Max" and self.endMarker.markerType == "Zero":
            self.segmentType = [1, 0]
        elif self.startMarker.markerType == "Max" and self.endMarker.markerType == "Min":
            self.segmentType = [1, -1]
        elif self.startMarker.markerType == "Min" and self.endMarker.markerType == "Max":
            self.segmentType = [-1, 1]
    
    def getSegmentBounds(self):
        return (self.startInd, self.endInd)
    
    def getSegmentRange(self):
        return range(self.startInd, self.endInd)
    
    def getSegmentType(self):
        return self.segmentType
    
    def plotSegmentBlindScatter(self, xVals, yVals, xLabel = "", yLabel = "", title = "", orbitTypeDict = None, colorVar = None):
        
        fig = plt.figure()
        axis = fig.add_subplot()

        colorMap = np.linspace(0, self.endInd-self.startInd, num=self.endInd-self.startInd)
        
        if not orbitTypeDict is None:
            fig.suptitle(str(title)+" Segment type: "+str(orbitTypeDict[self.segmentType]))
        else:
            fig.suptitle(str(title)+" Segment type: "+str(self.segmentType))

        axis.scatter(xVals[self.startInd:self.endInd], yVals[self.startInd:self.endInd], s=2, c=colorVar, cmap='copper')

        axis.set(xlabel = xLabel, ylabel=yLabel)

        plt.draw()


class Marker:
    def __init__(self, markerType, markerData):
        self.markerType = markerType
        self.ind = markerData
    
    def __repr__(self):
        return repr((self.markerType, self.ind))

class FullOrbit:
    # Takes an array of the local mins, local maxes, and zeros, respectively
    def __init__(self, data, indBounds = None, threshold = None, countZeros = False):
        if indBounds is None:
            self.indBounds = (0, len(data)-1)
        else:
            self.indBounds = indBounds
        
        self.data = data

        extrema = processpy.classifyExtrema(self.data, indBounds=indBounds, threshold=threshold)
        self.mins, self.maxes = extrema[0], extrema[1]

        if countZeros:
            self.markersPerOrbit = 4
            if threshold is None:
                self.zeros = processpy.findIntercepts(self.data, indBounds=self.indBounds)
            else:
                self.zeros = processpy.findIntercepts(self.data, indBounds=self.indBounds, yValue=threshold)
        else:
            self.markersPerOrbit = 2
            self.zeros = np.zeros(0)
        
        self.fullList = np.zeros(0)

        for min in self.mins:
            self.fullList = np.append(self.fullList, Marker("Min", min))
        for max in self.maxes:
            self.fullList = np.append(self.fullList, Marker("Max", max))
        for zero in self.zeros:
            self.fullList = np.append(self.fullList, Marker("Zero", zero))
        
        self.fullList = sorted(self.fullList, key=lambda marker: marker.ind)
    
    def getSegments(self, startInd):
        if startInd+self.markersPerOrbit >= len(self.fullList):
            raise ValueError("Cannot complete full orbit after marker index "+str(startInd)+" ("+str(startInd+1)
                             +"th marker). Total number of markers: "+str(len(self.fullList)))
        
        segmentList = np.zeros(0)

        for i in range(startInd, self.markersPerOrbit+startInd):
            segmentList = np.append(segmentList, OrbitSegment(self.fullList[i], self.fullList[i+1]))
        
        return segmentList
    
    def getSegment(self, ind):
        if ind+1 >= len(self.fullList):
            raise ValueError("Cannot get segment from marker index "+str(ind)+" to "+str(ind+1)+". Total number of markers: "+str(len(self.fullList)))
        return OrbitSegment(self.fullList[ind], self.fullList[ind+1])
    
    def getAllMarkers(self):
        return self.fullList
    
    def plotOrbitExtrema(self, xData = None, xLabel = "", yLabel = "", title = ""):
        
        if xData is None:
            xData = range(0, len(self.data))

        fig = plt.figure()
        axis = fig.add_subplot()

        fig.suptitle(title)

        axis.plot(xData[self.indBounds[0]:self.indBounds[1]], self.data[self.indBounds[0]:self.indBounds[1]])
        for theMarker in self.fullList:
            axis.plot(xData[theMarker.ind], self.data[theMarker.ind], "x")

        """xfmt = md.DateFormatter('%H:%M')
        axis.xaxis.set_major_formatter(xfmt)"""

        axis.set(xlabel = xLabel, ylabel = yLabel)

        plt.draw()