#takes article set and writes a file of tokens and features for the content span classifier

import verbCuesFeatureExtractor as verbCues
import contentSpanExtractor as contentSpans
import intermediaries.kNN as kNN
import pycrfsuite as crf
import sys
import os
import multiprocessing
from multiprocessing import Manager

data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../', 'data/'))
maxNumFiles = -1

#function takes the file, finds mathching annotated and raw files and calls VerbCues and then ContentSpans to 
#come up with the features
def workerFunction(myFile, listOfAnnotatedFiles, listOfRawFiles, return_list):


	filename = myFile.split('/')[-1]
	fileNoXML = filename.split('.xml')[0]

	print filename

	myAnnotatedFile = None

	#extract the PARC filename that match the title of the NLP filename 
	myAnnotatedFile = [s for s in listOfAnnotatedFiles if filename in s]
	myRawFile = [s for s in listOfRawFiles if fileNoXML in s][0]

	print myAnnotatedFile
	if len(myAnnotatedFile) == 1:
		myAnnotatedFile = myAnnotatedFile[0]
		flagNoLabels = True

	else:
		#didn't find a file
		print 'error opening Annotated File'
		return return_list
	
	try:
		verbsList, article = verbCues.openFile(myFile, myAnnotatedFile, myRawFile)
	except:
		raise
		print 'could not open'
	labelledVerbs = []

	for verb in verbsList:

		word = verb[0].split('=')[1]

		metaData = verb[-1].split('=')[1]
		metadata = metaData.split(';')
		sentID =  metadata[0]
		tokID = metadata[1]
		fileName = metadata[2]

		result = verb[-2].split('=')[1]
		#myfile.write(word + ',' + metaData + ',' + result + '\n')
		labelledVerbs.append([word, sentID, tokID, fileName, result])
	
	for verb in labelledVerbs:
		tokID = int(verb[2])
		sentID = int(verb[1])
		label = verb[4]
		if label == 'Y':
			article.sentences[sentID]['tokens'][tokID]['verbCue'] = True

	fileRows = contentSpans.findFeatures(filename, article, labelledVerbs, myAnnotatedFile)

	print (len(fileRows))
	return_list += fileRows


#to divide the files into list of lists
def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in xrange(0, len(l), n))

#multiprocesses the extraction
def train(pathToCORENLP, pathToAnnotatedFiles, pathToRaw, output):
	listOfNLPFiles = verbCues.openDirectory(pathToCORENLP)
	listOfAnnotatedFiles = verbCues.openDirectory(pathToAnnotatedFiles)
	listOfRawFiles = verbCues.openDirectory(pathToRaw)

	splitLists = list(chunks(listOfNLPFiles, len(listOfNLPFiles)/15))

	lastList = splitLists[-1]
	del splitLists[-1]

	lengthLists = len(splitLists[0])

	jobs = []
	manager = Manager()
	return_list = manager.list()


	#first lists are all equally sized, pick one from each at each iteration
	for i in range(lengthLists):
		#if i == 1:
		#	break
		for thisList in splitLists:
			myFile = thisList[i]
			p = multiprocessing.Process(target = workerFunction, args=(myFile, listOfAnnotatedFiles, listOfRawFiles, return_list))
			jobs.append(p)
			p.start()

	

	#append the files from last list (remainder of total files divided by 10)
	for myFile in lastList:
		p = multiprocessing.Process(target = workerFunction, args=(myFile, listOfAnnotatedFiles, listOfRawFiles, return_list))
		jobs.append(p)
		p.start()

	for proc in jobs:
		proc.join()

	open(os.path.join(data_dir, output), 'w').close()
	contentSpans.writeToTXT(return_list, os.path.join(data_dir, output), False)

	return return_list


#parse command line arguments
def main():
	usageMessage = '\nCorrect usage of the Training command is as follows: \n' + \
					'python source/trainData.py /pathToCoreNLPDirectory /pathToParc nameOfOutputFile.txt \n' + \
					'\nFor reference, the path to the CoreNLP file is: /home/ndg/dataset/ptb2-corenlp/CoreNLP/ + train, test or dev depending on your needs. \n' + \
					'\n The path to raw files are: /home/ndg/dataset/ptb2-corenlp/masked_raw/ + train, test, dev' + \
					'The path to the Parc3 files is /home/ndg/dataset/parc3/ + train, test or dev depending on your needs.\n'

	args = sys.argv

	if len(args) == 5:

		pathToCORENLP = args[1]
		pathToAnnotatedFiles = args[2]
		pathToRawFiles = args[3]

		output = args[4]

		if os.path.isdir(pathToCORENLP):
			print 'valid path to a directory'
		else:
			print 'ERROR: The path to this coreNLP directory does not exist.'
			print usageMessage
			return

		if os.path.isdir(pathToAnnotatedFiles):
			print 'valid path to a directory'
		else:
			print 'ERROR: The path to this annotated file directory does not exist.'
			print usageMessage
			return

		if os.path.isdir(pathToAnnotatedFiles):
			print 'valid path to a directory'
		else:
			print 'ERROR: The path to this annotated file directory does not exist.'
			print usageMessage
			return


		train(pathToCORENLP, pathToAnnotatedFiles, pathToRawFiles, output)


	else:
		print usageMessage



if __name__ == '__main__':
   main()