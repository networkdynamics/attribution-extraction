#accepts either a file with verb cue features or a list of features and assigns them a "y" or 'N' label

import csv
import os
import sys
import math
import operator
from time import strftime
import timeit
import cProfile
import numpy as np


data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'data/'))

#opens a file and returns all the rows
def readData(myfile, flagTraining):


	rowsOfData = []
	countInside = 0
	countOutside = 0
	countColon = 0
	myfile = os.path.join(data_dir, myfile)
	with open(myfile, 'r') as myfile:
		reader = csv.reader(myfile, delimiter=',')
		#if flagTraining == True:
		#	reader = subsampleFile(reader)
		for row in reader:
			row = processRow(row)
			rowsOfData.append(row)

	myfile.close()




	return rowsOfData

def subsampleFile(myfile):
	yesses = []
	noes = []
	for verb in myfile:
		label = verb[-2]
		if label == 'label=Y':
			yesses.append(verb)
		elif label == 'label=N':
			noes.append(verb)

	lengthYes = len(yesses)
	fortyfivePervent = round((lengthYes*100)/45) - lengthYes

	newNoesIndices = np.random.choice(len(noes), fortyfivePervent, replace=False)

	newNoes = []
	for index in newNoesIndices:
		newNoes.append(noes[index]) 

	return yesses+newNoes

#reformats the row so that this program can read it (e.g. list of classes / siblings)
def processRow(row):

	for indx, elem in enumerate(row):
		elem = elem.split('=', 1)[1]
		if indx != 18:
			row[indx] = elem.lower()
		else:
			row[indx] = elem

	verbNetClasses = row[11]
	classes = verbNetClasses.split(';!')
	for indx, oneClass in enumerate(classes):
		oneClass = oneClass.split('-', 1)[0]
		classes[indx] = oneClass
	row[11] = classes

	siblingNodes = row[14]
	siblings = siblingNodes.split(';!')
	row[14] = siblings

	label = row[18].upper()
	metadata = row[19]

	row[12] = int(row[12])
	row[15] = int(row[15])
	row[16] = int(row[16])

	row = row[:19]

	return row, metadata

#distance calculations
def intDistance(int1, int2):
	if int1 == int2:
		return 0.0
	elif int1 + int2 == 0:
		return 0.0
	return abs(float(int2 - int1))/(int2 + int1)

#list distance
def listDistance(list1, list2):
	#uniqueVal = set(list1 + list2)
	#sameVals = list(set(list1).intersection(list2))


	if list1 == list2:
		return 0
	elif (list1 == [''] or list2 == ['']):
		return 1
	else:
		allClasses = list(set(list1 + list2))

		similar = (len(list1) + len(list2)) - len(allClasses)
		return 1.0 - float(similar)/len(allClasses)

#total distance
def calculateDistance(vec1, vec2, fourthMax, normalizeValDepth):

	distanceFINAl = 0

	#if first lemma is not the same, then first word is also not the same
	if (vec1[1] != vec2[1]):
		distanceFINAl += 1
		distanceFINAl += 2
	elif (vec1[0] != vec2[0]):
		distanceFINAl += 1
	if distanceFINAl > fourthMax:
		return 1000

	#first POS similarity
	if (vec1[2] != vec2[2]):
		distanceFINAl += 1
	
	#second lemma
	if (vec1[4] != vec2[4]):
		distanceFINAl += 1
		distanceFINAl += 2
	elif (vec1[3] != vec2[3]):
		distanceFINAl += 1

	#second POS similarity
	
	if (vec1[5] != vec2[5]):
		distanceFINAl += 1

	if distanceFINAl > fourthMax:
		return 1000


	#third lemma
	if (vec1[7] != vec2[7]):
		distanceFINAl += 1
		distanceFINAl += 2


	elif (vec1[6] != vec2[6]):
		distanceFINAl += 1

	if distanceFINAl > fourthMax:
		return 1000
	#third POS
	if (vec1[8] != vec2[8]):
		distanceFINAl += 1

	if distanceFINAl > fourthMax:
		return 1000

	if (vec1[9] != vec2[9]):
		distanceFINAl += 1

	if distanceFINAl > fourthMax:
		return 1000


	if (vec1[10] != vec2[10]):
		distanceFINAl += 1

	if distanceFINAl > fourthMax:
		return 1000

	if (vec1[13] != vec2[13]):
		distanceFINAl += 1

	if distanceFINAl > fourthMax:
		return 1000

	if (vec1[17] != vec2[17]):
		distanceFINAl += 10

	if distanceFINAl > fourthMax:
		return 1000

	if vec1[15] + vec1[16] == 0:
		quadrantVec1 = 0
	else:
		quadrantVec1 = float(vec1[15])/(vec1[15] + vec1[16])

	if vec2[15] + vec2[16] == 0:
		quadrantVec2 = 0
	else:
		quadrantVec2 = float(vec2[15])/(vec2[15] + vec2[16])


	quadrantDistance = (abs(quadrantVec1 - quadrantVec2))

	distanceFINAl += quadrantDistance

	if distanceFINAl > fourthMax:
		return 1000

	depthDistNormal = intDistance(vec1[12], vec2[12])/normalizeValDepth

	if depthDistNormal > 1.0:
		distanceFINAl += 1
	else:
		distanceFINAl += intDistance(vec1[12], vec2[12])/normalizeValDepth
		
	if distanceFINAl > fourthMax:
		return 1000


	distanceFINAl += listDistance(vec1[14], vec2[14])

	if distanceFINAl > fourthMax:
		return 1000

	distanceFINAl += listDistance(vec1[11], vec2[11])

	return distanceFINAl

def column(matrix, i):
    return [row[i] for row in matrix]

#calculate all the distances, create neighbor array
def getNeighbors(trainingSet, test, k):
	distances = []

	topMax = ('', 100)
	secondMax = ('', 100)
	thirdMax = ('', 100)
	fourthMax = ('', 100)

	thisDepth = test[12]
	depthColumn = column(trainingSet, 12)
	depths = sorted(depthColumn)

	maxIndex = int(round(0.95 * len(depths)))
	minIndex = int(round(0.05 * len(depths)))

	maxDepth = depths[maxIndex]
	minDepth = depths[minIndex]

	normalizeValDepth = max(abs(float(maxDepth - thisDepth)/(maxDepth + thisDepth)), abs(float(minDepth - thisDepth)/(minDepth + thisDepth)))

	for indx in range(len(trainingSet)):
		distance = calculateDistance(test, trainingSet[indx], fourthMax[1], normalizeValDepth)
		#print trainingSet[indx]
		if distance == 1000:
			continue
		if distance < topMax[1]:
			fourthMax = thirdMax
			thirdMax = secondMax
			secondMax = topMax
			topMax = ((trainingSet[indx], distance))
		elif distance < secondMax[1]:
			fourthMax = thirdMax
			thirdMax = secondMax
			secondMax = ((trainingSet[indx], distance))
		elif distance < thirdMax[1]:
			fourthMax = thirdMax
			thirdMax = ((trainingSet[indx], distance)) 
		elif distance < fourthMax[1]:
			fourthMax = ((trainingSet[indx], distance)) 

	neighbors = []

	neighbors.append(topMax)
	neighbors.append(secondMax)
	neighbors.append(thirdMax)
	neighbors.append(fourthMax)



	return neighbors

#use majority vote to get the response
def getResponse(neighbors):
	classVotes = {}
	for x in range(len(neighbors)):
		response = neighbors[x][0][-1]
		if response in classVotes:
			classVotes[response] += 1
		else:
			classVotes[response] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	if len(sortedVotes) == 2 and sortedVotes[0][1] == sortedVotes[1][1]:
		return 'N'

	return sortedVotes[0][0]

#calculates accuracy of predictions vs. test set labels
def getAccuracy(testSet, predictions):
	correct = 0
	correctYes = 0
	totalYesPrediction = 0;
	totalYesReal = 0

	for x in range(len(testSet)):
		if predictions[x] == 'Y':
			totalYesPrediction += 1

		if testSet[x][-1] == predictions[x]:
			correct += 1

		if testSet[x][-1] == 'Y':
			totalYesReal += 1
			if predictions[x] == 'Y':
				correctYes += 1

	print 'precision'
	print float(correctYes)/totalYesPrediction
	print 'recall'
	print float(correctYes)/totalYesReal
	print 'accuracy'
	print float(correct)/len(predictions) * 100


	return (correct/float(len(testSet))) * 100.0

#saves the accuracy to a text file
def saveResults(accuracy, length):
	myfile = open(data_dir + '/knnExperiments.txt', 'a')
	myfile.write('On Date: ' + strftime("%Y-%m-%d %H:%M:%S") + ' Ariane trained the verbCue classifier' + \
		' with accuracy: ' + str(accuracy) + ' with a training set of length: ' + str(length) + '\n')
	myfile.close()

#for when the test set comes with labels
def test(myTrainingFile, myTestingFile):
	#data comes in as a list of tuples, [features of verb], metadata)
	trainingData = readData(myTrainingFile, True)
	testData = readData(myTestingFile, False)

	trainingSet, metaData = zip(*trainingData)
	testSet, metaDataTest = zip(*testData)

	print 'length of trainingSet: ' + str(len(trainingSet))
	print ' length of Test Set: ' + str(len(testSet))

	i = 0
	for item in trainingSet:
		if item[-1] == 'Y':
			i = i + 1
	print 'number of verb cues in training set ' + str(i)

	i = 0
	for item in testSet:
		if item[-1] == 'Y':
			i = i + 1
	print 'number of verb cues in test set ' + str(i)


	predictions = []

	k = 4
	print k

	#myfile = open(os.path.join(data_dir, 'TrainKNNVerbPredictions.csv'), 'w').close()
	#myfile = open(os.path.join(data_dir, 'TrainKNNVerbPredictions.csv'), 'a')

	for x in range(len(testSet)):
		word = testData[x][0][1]
		sentID = testData[x][1].split(';')[0]
		tokID = testData[x][1].split(';')[1]
		filename = testData[x][1].split(';')[2]
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		predictions.append(result)
		#myfile.write(word + ',' + sentID + ',' + tokID + ',' + filename + ',' + result + '\n')
		#if testSet[x][-1] != result:
		#	print testSet[x]
		#	print neighbors
		if (x == round(len(testSet)/4)):
			print '''25% done'''
		elif (x == round(len(testSet)/10)):
			print '''10% done'''
		elif (x == round(len(testSet)/2)):
			print "50% done"
		elif (x == round(len(testSet)/1.3)):
			print "75% done"
		#print('> ' + str(x) + ' predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy = getAccuracy(testSet, predictions)
	print('Accuracy: ' + repr(accuracy) + '%')

	saveResults(accuracy, len(trainingSet))

#takes rows, labels them, and returns them, takes in trainingData to use
def labelWithoutCSV(trainingData, labelRows):

	testData = []

	for row in labelRows:
		row = processRow(row)
		testData.append(row)

	trainingSet, metaData = zip(*trainingData)
	testSet, metaDataTest = zip(*testData)

	#only use first 12,000 rows for speed reasons
	#trainingSet = trainingSet[-6000:]
	#trainingSet = trainingSet[12000:-10000]

	print 'length of trainingSet: ' + str(len(trainingSet))
	print ' length of Test Set: ' + str(len(testSet))

	i = 0
	for item in trainingSet:
		if item[-1] == 'Y':
			i = i + 1
	print 'number of verb cues in training set ' + str(i)

	predictions = []

	k = 4
	print k

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
			print '''25% done'''
		elif (x == round(len(testSet)/10)):
			print '''10% done'''
		elif (x == round(len(testSet)/2)):
			print "50% done"
		elif (x == round(len(testSet)/1.3)):
			print "75% done"

	print 'Number of Verb Cues Found: ' + str(numberVerbCuesFound)
	return resultArray


#label using files, training and labeling file
def label(myTrainingFile, myLabelingFile):
	labelingOutput = myLabelingFile.split('.csv')[-2] + 'Labelled.csv'
	#labelingOutput = labelledFile.split('.csv')[0] + 'Labeled.csv'
	print labelingOutput

	#empty this file
	myfile = open(os.path.join(data_dir, labelingOutput), 'w').close()
	myfile = open(os.path.join(data_dir, labelingOutput), 'a')

	trainingData = readData(myTrainingFile)
	testData = readData(myLabelingFile)

	trainingSet, metaData = zip(*trainingData)
	testSet, metaDataTest = zip(*testData)

	#only use first 12,000 rows for speed reasons
	#trainingSet = trainingSet[:12000]

	print 'length of trainingSet: ' + str(len(trainingSet))
	print ' length of Test Set: ' + str(len(testSet))

	i = 0
	for item in trainingSet:
		if item[-1] == 'Y':
			i = i + 1
	print 'number of verb cues in training set ' + str(i)

	predictions = []

	k = 4
	print k

	numberVerbCuesFound = 0

	for x in range(len(testSet)):
		neighbors = getNeighbors(trainingSet, testSet[x], k)
		result = getResponse(neighbors)
		myfile.write(testSet[x][0] + ',' + metaDataTest[x] + ',' + result + '\n')
		if result == 'Y':
			numberVerbCuesFound += 1
		if (x == round(len(testSet)/4)):
			print '''25% done'''
		elif (x == round(len(testSet)/10)):
			print '''10% done'''
		elif (x == round(len(testSet)/2)):
			print "50% done"
		elif (x == round(len(testSet)/1.3)):
			print "75% done"

	myfile.close()
	print 'Number of Verb Cues Found: ' + str(numberVerbCuesFound)


#parse command line
def main():
	usageMessage = '\nCorrect usage of the kNN classifier command is as follows: \n\n' + \
					'To test a labelled dataset: \n python source/intermediaries/kNN.py -test /pathToTrainingCSVFile /pathToTestCSVFile \n' + \
					'\t If no path is provided, we default to data/PARCTrainVerbFeats.csv and data/PARCdevVerbFeats.csv. This data was generated from the PARC3 training files. \n \n' + \
					'To label a dataset: \n python source/intermediaries/kNN.py -label /pathToTrainingCSVFile /pathToUnlabelledCSVFile \n' + \
					'This will create a csv file with a \'Labeled\' tag appended to the name with the predicted labels. \n' + \
					'\t If no paths are provided, we default to data/PARCTrainVerbFeats.csv and data/PARCdevVerbFeats.csv. This data was generated from the PARC3 test and train files. \n' + \
					'\nProperly formatted data is generated from running XML files through the verbCuesFeatExtractor.py program\n' 

	args = sys.argv


	if len(args) == 4:

		pathToTraining = args[2]
		pathToTestCSVFile = args[3] 
		print pathToTraining
		print pathToTestCSVFile

		if os.path.isfile(os.path.join(data_dir, pathToTraining)):
			print 'valid path to a file'
		else:
			print 'ERROR: The path to this training file does not exist.'
			return

		if os.path.isfile(os.path.join(data_dir, pathToTestCSVFile)):
			print 'valid path to a file'
		else:
			print 'ERROR: The path to this testing file does not exist.'
			return

		if args[1] == '-test':
			print 'testing'

			pr = cProfile.Profile()
			pr.enable()
			test(pathToTraining, pathToTestCSVFile)
			pr.disable()
			pr.print_stats(sort='time')

		elif args[1] == '-label':
			print 'labelling'
			label(pathToTraining, pathToTestCSVFile)
		else:
			print usageMessage

	elif len(args) == 2:
		if args[1] == '-test':
			print 'training'
			print 'defaulting to the data from data/PARCTrainVerbFeats.csv and data/PARCdevVerbFeats.csv, which was generated from the PARC3 training and dev dataset'
			start = timeit.default_timer()

			pr = cProfile.Profile()
			pr.enable()
			test(data_dir + '/PARCTrainVerbFeats.csv', data_dir + '/PARCTestVerbFeats.csv')
			pr.disable()
			pr.print_stats(sort='time')


			stop = timeit.default_timer()

			print 'time spend: ' + str(stop - start) 
			
		elif args[1] == '-label':
			print 'defaulting to the data from data/PARCTrainVerbFeats.csv and data/PARCdevVerbFeats.csv, which was generated from the PARC3 test dataset'
			print 'labelling'
			label(data_dir + '/PARCTrainVerbFeats.csv', data_dir + '/PARCdevVerbFeats.csv')


		else:
			print usageMessage
	else:
		print usageMessage


if __name__ == '__main__':
   main()