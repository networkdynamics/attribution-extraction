import numpy as numpy
from sklearn import preprocessing
import csv
import os
import sys
import math
import operator
from time import strftime
import timeit
from sklearn import preprocessing
import itertools

def encodeAllRows(rows):
	print rows[0]

	encoders = []


	#verbNetClasses = [r[0].pop(12) for r in rows]
	
	#grammar = [r[0].pop(14) for r in rows]

	#rows = [r[0] for r in rows]

	#rows = numpy.array(rows)

	reformedRows = []
	OGLength = len(rows[0][0])
	for i in range(OGLength):
		currentRow = []
		for row, metadata in rows:
			currentRow.append(row[i])
		reformedRows.append(currentRow)


	for elem in reformedRows:
		if isinstance(elem[0], list):
			flatList = list(itertools.chain.from_iterable(elem))
			encoder = preprocessing.LabelEncoder()
			encoder.fit(flatList)
			print encoder.classes_
			encoders.append(encoder)
			continue


		encoder = preprocessing.LabelEncoder()
		encoder.fit(elem)
		print encoder.classes_
		encoders.append(encoder)

	print len(encoders)
	return encoders

def encodeTestRows(testData, encoders):
	finalRows = []
	print testData[0]
	for row, metadata in testData:
		thisRow = []
		for i, item in enumerate(row):
			if i == 12 or i == 15:
				currentEncoder = encoders[i]
				encoded = currentEncoder.transform(item)
				thisRow.append(encoded)
			else:
				currentEncoder = encoders[i]
				encoded = currentEncoder.transform([item])
				thisRow.append(encoded)
		finalRows.append(thisRow)

	print testData[0]
	print finalRows[0]



	return finalRows


def processRow(row):


	for indx, elem in enumerate(row):
		elem = elem.split('=', 1)[1]
		row[indx] = elem

	verbNetClasses = row[12]
	classes = verbNetClasses.split(';!')
	for indx, oneClass in enumerate(classes):
		oneClass = oneClass.split('-', 1)[0]
		classes[indx] = oneClass
	row[12] = classes

	siblingNodes = row[15]
	siblings = siblingNodes.split(';!')
	row[15] = siblings

	label = row[19]
	metadata = row[20]

	row[13] = int(row[13])
	row[16] = int(row[16])
	row[17] = int(row[17])

	row = row[:19]

	return row, metadata

def labelWithoutCSV(trainingData, labelRows):

	testData = []

	for row in labelRows:
		row = processRow(row)
		testData.append(row)

	trainingLabels = [r[0].pop(-1) for r in trainingData]
	trainlabelsMapped = []
	for elem in trainingLabels:
		if elem == 'Y':
			trainlabelsMapped.append(1)
		elif elem == 'N':
			trainlabelsMapped.append(0)
		else:
			raise


	print trainingData[0]
	print testData[0]

	allRows = trainingData + testData
	encoders = encodeAllRows(allRows)

	testRows = encodeTestRows(testData, encoders)
	trainRows = encodeTestRows(trainingData, encoders)







	#trainingSet, metaData = zip(*trainingData)
	#testSet, metaDataTest = zip(*testData)



'''
	#only use first 12,000 rows for speed reasons
	trainingSet = trainingSet[:12000]

	print 'length of trainingSet: ' + str(len(trainingSet))
	print ' length of Test Set: ' + str(len(testSet))

	i = 0
	for item in trainingSet:
		if item[-1] == 'Y':
			i = i + 1
	print 'number of verb cues in training set ' + str(i)

	predictions = []

	k = 4

	numberVerbCuesFound = 0

	resultArray = []

	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)

		metadata = metaDataTest[x].split(';')
		sentID =  metadata[0]
		tokID = metadata[1]
		fileName = metadata[2]

		resultArray.append([testSet[x][0], sentID, tokID, fileName, result])

		if result == 'Y':
			numberVerbCuesFound += 1
		if (x == round(len(testSet)/4)):
			print "25% done"
		elif (x == round(len(testSet)/10)):
			print "10% done"
		elif (x == round(len(testSet)/2)):
			print "50% done"
		elif (x == round(len(testSet)/1.3)):
			print "75% done"

	print 'Number of Verb Cues Found: ' + str(numberVerbCuesFound)
	return resultArray
'''
