import numpy as np

# takes in a 2D data array, a 1D array of x-values, and optionally interpolation bounds
# and returns the data y-values interpolated onto the input x-values
# and an array of the index values of the x-values
def interpolate(data, xValues, xBounds = None):

    # takes in two points and a t value, and returns the linearly interpolated y-value for the t-value
    def linInt(pointA, pointB, x):
        dy = pointB[1] - pointA[1]
        dtFull = pointB[0] - pointA[0]
        dtInner = x-pointA[0]
        y = dy*(dtInner/dtFull)+pointA[1]
        return y
    
    if xBounds is None:
        xBounds = (xValues[0], xValues[-1])
    dataX = data[0]
    dataY = data[1]

    if len(dataX) != len(dataY):
        print(f'''processpy.py WARNING: The data arrays are not the same length: x has {str(len(dataX))} values and y has {str(len(dataY))}
              . This can cause an out of bounds error.''')
        

    interpolatedData = np.zeros(0)
    
    if len(data[0]) < 2:
        raise ValueError('The data must have at least 2 values for interpolation, but only has '+str(len(dataX))+' data value(s).')

    indData = 1
    # indData starts at 1, and the interpolation is between indData-1 and indData
    indX = 0

    #
    if xBounds[-1] < xValues[0]:
        raise ValueError('The selected end boundary of '+str(xBounds[-1])+' is earlier than the first interpolation value of '+str(xValues[0])+'.')

    # if the timestamps time starts out less than the start time
    while xValues[indX] < xBounds[0]:
        indX += 1
        if indX >= len(xValues):
            raise ValueError('The selected start boundary of '+str(xBounds[0])+' is later than the final interpolation value of '+str(xValues[-1])+'.')
    
    # indXStart for determining the range of x-values
    indXStart = indX

    # if the if the x-value of the second data point (for interpolation) is less than the first value of xValues
    while dataX[indData] < xValues[indX]:
        indData += 1
        if indData >= len(dataX):
            print('processpy.py WARNING: The data range from '+str(dataX[1])+' to '+str(dataX[-1])+' does not overlap with the interpolation range from '
                        +str(xValues[indXStart])+' to '+str(xValues[-1])+'. All values are set to nan.')
            break


    # if the timestamps time starts out less than the data time it will fill those values with nan
    while indX < len(xValues) and xValues[indX] < dataX[indData - 1]:
        indX += 1
        interpolatedData = np.append(interpolatedData, np.nan)
    
    if indX >= len(xValues):
            print('processpy.py WARNING: The data range from '+str(dataX[1])+' to '+str(dataX[-1])+' does not overlap with the interpolation range from '
                        +str(xValues[indXStart])+' to '+str(xValues[-1])+'. All values are set to nan.')
    
    while indX < len(xValues) and xValues[indX] <= xBounds[1]:
        # Pads the data with nan values if it is shorter than the interpolation range
        if indData >= len(dataX):
            interpolatedData= np.append(interpolatedData, np.nan)
            indX += 1
        else:
            if xValues[indX] > dataX[indData]:
                indData += 1
            else:
                newYVal = linInt((dataX[indData-1], dataY[indData-1]), (dataX[indData], dataY[indData]), xValues[indX])
                interpolatedData= np.append(interpolatedData, newYVal)
                indX += 1

    if indX >= len(xValues):
        indXEnd = len(xValues)
    else:
        indXEnd = indX
    
    return(interpolatedData, range(indXStart, indXEnd))


# takes a sequentially sorted array and returns the index of the key within that array
# if the key is between two values, it returns the lower value
def binarySearch(sortedArrayFull, key, indBounds = None, ascending = True): 
    
    if indBounds is None:
        indMin, indMax = 0, len(sortedArrayFull)-1
    
    else:
        indMin, indMax = indBounds[0], indBounds[1]
    
    sortedArray = sortedArrayFull[indMin:indMax+1]

    if len(sortedArray) < 1:
        raise ValueError('Cannot do binary search on an array with 0 values.')
    
    low = 0
    high = len(sortedArray)-1
    ind = int(high/2)
    while low <= high:
        if key == sortedArray[ind]:
            return ind
        elif ascending:
            if key > sortedArray[ind]:
                low = ind + 1
            elif key < sortedArray[ind]:
                high = ind - 1
        elif not ascending:
            if key < sortedArray[ind]:
                low = ind + 1
            elif key > sortedArray[ind]:
                high = ind - 1
        ind = int((high+low)/2)

    if ind == 0 and key < sortedArray[ind]:
        print('processpy.py WARNING: The key '+str(key)+' is less than the range of the sorted array (min value: '+str(sortedArray[ind])
        +'). Returned index '+str(ind)+'.')
    elif ind == len(sortedArray)-1 and key > sortedArray[ind]:
        print('processpy.py WARNING: The key '+str(key)+' is greater than the range of the sorted array (max value: '+str(sortedArray[ind])
        +'). Returned index '+str(ind)+'.')
    
    return ind+indMin

# Takes a vector input and returns the vector magnitude as a scalar
def magnitude(vector):
    sum = 0
    for comp in vector:
        sum += comp**2
    return np.sqrt(sum)


# Takes 1D array and returns the intercept indeces with the yValue variable
def findIntercepts(data, indBounds = None, yValue = 0):
    
    if indBounds is None:
        indMin, indMax = 0, len(data)-1
    
    else:
        indMin, indMax = indBounds[0], indBounds[1]
    
    zeros = np.zeros(0)

    isPositivePast = data[indMin] > yValue

    for i in range(indMin, indMax):
        isPositiveNow = data[i] > yValue

        if (isPositivePast != isPositiveNow and data[i-1] != yValue) or data[i] == yValue:
            zeros = np.append(zeros, int(i))
        
        isPositivePast = isPositiveNow
    
    return zeros.astype(int)

# Takes 1D array and returns the minima and maxima
# indBounds sets bounds for the indices
# threshold sets a value above which no minima are taken and below which no maxima are taken.
# If a threshold value is given there will only be one maximum/minimum per section that the graph
# crosses the threshold.
def classifyExtrema(data, indBounds = None, threshold = None):
    
    # Parses through the data step by step to locate local extrema
    # by finding places were the 
    def stepWise():
        index = indMin+1

        isIncreasingPast = data[index] > data[index-1]
        index += 1

        maxes = np.zeros(0)
        mins = np.zeros(0)

        while index < indMax:
            isIncreasingNow = data[index] > data[index-1]
            # Local Max
            if isIncreasingPast and not isIncreasingNow:
                maxes = np.append(maxes, index)
            
            # Local Min
            if not isIncreasingPast and isIncreasingNow:
                mins = np.append(mins, index)
            
            isIncreasingPast = isIncreasingNow
            index += 1
        
        return (mins.astype(int), maxes.astype(int))
    
    def chunkWise():

        zeros = findIntercepts(data, indBounds=(indMin+1, indMax-1), yValue=threshold)
        zeros = np.append(indMin, zeros)
        zeros = np.append(zeros, indMax)

        index = 1

        maxes = np.zeros(0)
        mins = np.zeros(0)

        while index < len(zeros):
            chunk = data[zeros[index-1]:zeros[index]]
            maxLoc = np.argmax(chunk) + zeros[index-1]
            minLoc = np.argmin(chunk) + zeros[index-1]

            if data[maxLoc] > threshold and maxLoc != indMin and maxLoc != indMax-1:
                maxes = np.append(maxes, maxLoc)

            elif data[maxLoc] < threshold and minLoc != indMin and minLoc != indMax-1:
                mins = np.append(mins, minLoc)
            
            index += 1

        return (mins.astype(int), maxes.astype(int))


    if indBounds is None:
        indMin, indMax = 0, len(data)-1
    
    else:
        indMin, indMax = indBounds[0], indBounds[1]
    

    if (indMax-indMin) < 2:
        raise ValueError('The data must have at least 3 values for extrema, but only has '+str(indMax-indMin)+' data value(s).')
    
    if threshold is None:
        return stepWise()
    
    return chunkWise()

# Takes in two arrays of points and uses brute force technique to return
# a 2 by N array of the index in each list for each of the N intersections
# Note: always returns the earlier index of intersection
def intersect2DQuadratic(x1, y1, x2, y2):
    intersections = np.zeros((0, 2))
    for i1 in range(1, len(x1)):
        for i2 in range(1, len(x2)):
            if pointIntersection((x1[i1-1], y1[i1-1]), (x1[i1], y1[i1]), (x2[i2-1], y2[i2-1]), (x2[i2], y2[i2])):
                intersections = np.append(intersections, [[i1-1, i2-1]], axis=0)
    return intersections

# takes in 4 points and returns whether the first two points
# produce a line segment with which the segment between the second two points
# intersects
def pointIntersection(points1A, points1B, points2A, points2B):

    slope1 = (points1B[1]-points1A[1])/(points1B[0]-points1A[0]) # simple rise/run
    slope2 = (points2B[1]-points2A[1])/(points2B[0]-points2A[0])

    if slope1 == slope2:
        return False
    
    intercept1 = slope1*(0-points1A[0])+points1A[1] # point-slope
    intercept2 = slope2*(0-points2A[0])+points2A[1]

    intersectionX = (intercept2-intercept1)/(slope1-slope2) # m1x + b1 = m2x + b2

    inBounds1 = (intersectionX >= points1A[0] and intersectionX <= points1B[0]) or (intersectionX >= points1B[0] and intersectionX <= points1A[0])
    inBounds2 = (intersectionX >= points2A[0] and intersectionX <= points2B[0]) or (intersectionX >= points2B[0] and intersectionX <= points2A[0])
    
    return inBounds1 and inBounds2

# Works by drawing n lines and seeing if they intersect
def intersect2D(x1, y1, x2, y2, n = None):
    if n is None:
        n = int(min(len(x1), len(x2))/4)
    intersections = np.zeros((0, 2))
    if n >= len(x1)-1 and n >= len(x2)-1:
        return intersect2DQuadratic(x1, y1, x2, y2).astype(int)
    
    indices1 = np.linspace(0, len(x1)-1, n).astype(int)
    indices2 = np.linspace(0, len(x2)-1, n).astype(int)
    for i1 in range(1, len(indices1)):
        for i2 in range(1, len(indices2)):
            s1Start = indices1[i1-1]
            s1End = indices1[i1]
            s2Start = indices2[i2-1]
            s2End = indices2[i2]
            if pointIntersection((x1[s1Start], y1[s1Start]), (x1[s1End], y1[s1End]), (x2[s2Start], y2[s2Start]), (x2[s2End], y2[s2End])):
                intersections = np.append(intersections, intersect2D(x1[s1Start:s1End+1], y1[s1Start:s1End+1], 
                                                                     x2[s2Start:s2End+1], y2[s2Start:s2End+1], n=n) + [s1Start, s2Start], axis=0)
    return intersections.astype(int)

    

# Takes in x and y and returns a 3D array of linear segments
# This is useful for plotting a line with a color gradient in matplotlib
def getColorSegments(x, y):
    points = np.append(np.transpose([x]), np.transpose([y]), axis=1)

    segments = np.zeros((len(x)-1, 2, 2))
    for i in range(0, len(points)-1):
        segments[i, :, :] = np.append([points[i]], [points[i+1]], axis=0)
    
    return segments