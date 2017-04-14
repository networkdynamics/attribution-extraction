#takes an article, finds heads of verbs and extracts all features

print 'importing'
from nltk.tree import *
from nltk.corpus import verbnet as vn
import csv
import os
import sys
from parc_reader import ParcCorenlpReader as P
import multiprocessing
from multiprocessing import Manager
print 'imports done'
import time
import numpy as np

maxNumFiles = -1
minNumFile = 0
data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/'))

#gets datapath, creates a list of files and any nested files (only goes down by one subdirectory)
def openDirectory(datapath):
	listOfFiles = []

	for item in os.listdir(datapath):
		if os.path.isdir(os.path.join(datapath, item)):
			item = os.path.join(datapath, item)
			for newItem in os.listdir(item):
				newItem = os.path.join(item, newItem)
				listOfFiles.append(newItem)
		elif os.path.isfile(os.path.join(datapath, item)):
			item = os.path.join(datapath, item)
			listOfFiles.append(item)

	return listOfFiles

#write to a csv output file as designated by command line
def writeToTSV(rows, outputFile):

	with open(outputFile, 'w') as myfile:
		writer = csv.writer(myfile, delimiter=',')
		writer.writerows(rows)
	myfile.close()

	print '\nData written to ' + outputFile + '\n'

#to multiprocess this stuff
def workerFunction(myFile, listOfAnnotatedFiles, listOfRawFiles, flagNoLabels, return_list):

	filename = myFile.split('/')[-1]
	fileNoXML = filename.split('.xml')[0]


	#print filename

	myAnnotatedFile = None

	#this means there is an annotated file list, error if no corresponding file is found
	if flagNoLabels == False:
		myAnnotatedFile = [s for s in listOfAnnotatedFiles if filename in s]
		myRawFile = [s for s in listOfRawFiles if fileNoXML in s][0]

		if len(myAnnotatedFile) == 1:
			myAnnotatedFile = myAnnotatedFile[0]
		else:
			print 'error opening Annotated File. There is probably no matching annotated file'
			j = j + 1
			return 

	rows, article = openFile(myFile, myAnnotatedFile, myRawFile)
	print filename

	return_list += rows

#divides list of files into list of lists
def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in xrange(0, len(l), n))

#split lists and then calls the multiprocessor
#get the files, open them, extract verbs and features and create a large array of rows

def findFiles(listOfNLPFiles, listOfAnnotatedFiles, listOfRawFiles, output):
	if listOfAnnotatedFiles == None:
		flagNoLabels = True
	else:
		flagNoLabels = False

	splitLists = list(chunks(listOfNLPFiles, len(listOfNLPFiles)/10))

	lastList = splitLists[-1]
	del splitLists[-1]

	lengthLists = len(splitLists[0])

	jobs = []
	manager = Manager()
	return_list = manager.list()


	j = 0
	#first lists are all equally sized, pick one from each at each iteration
	for i in range(lengthLists):
		#if i == 1:
		#	break
		for thisList in splitLists:
			myFile = thisList[i]
			p = multiprocessing.Process(target = workerFunction, args=(myFile, listOfAnnotatedFiles, listOfRawFiles, flagNoLabels, return_list))
			jobs.append(p)
			p.start()
		time.sleep(3)

	

	#append the files from last list (remainder of total files divided by 10)
	for myFile in lastList:
		p = multiprocessing.Process(target = workerFunction, args=(myFile, listOfAnnotatedFiles, listOfRawFiles, flagNoLabels, return_list))
		jobs.append(p)
		p.start()

	for proc in jobs:
		proc.join()


	open(os.path.join(data_dir, output), 'w+').close()

	#subListed = subsample(return_list)

	#for training set
	#writeToTSV(subListed, os.path.join(data_dir, output))

	#for testing sets
	writeToTSV(return_list, os.path.join(data_dir, output))


def subsample(listVerbs):
	yesses = []
	noes = []
	for verb in listVerbs:
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


#non-multiprocessing option
#get the files, open them, extract verbs and features and create a large array of rows
def findFiles1(listOfNLPFiles, listOfAnnotatedFiles, listOfRawFiles, output):

	if listOfAnnotatedFiles == None:
		flagNoLabels = True
	else:
		flagNoLabels = False

	myRows = []

	j = 0
	#open each NLP File
	for myFile in listOfNLPFiles:

		if j < minNumFile:
			j = j + 1
			continue

		files = len(listOfNLPFiles)
		filename = myFile.split('/')[-1]
		fileNoXML = filename.split('.xml')[0]

		print filename

		myAnnotatedFile = None

		#this means there is an annotated file list, error if no corresponding file is found
		if flagNoLabels == False:
			myAnnotatedFile = [s for s in listOfAnnotatedFiles if filename in s]
			myRawFile = [s for s in listOfRawFiles if fileNoXML in s][0]

			if len(myAnnotatedFile) == 1:
				myAnnotatedFile = myAnnotatedFile[0]
			else:
				print 'error opening Annotated File. There is probably no matching annotated file'
				j = j + 1
				continue

		print('opening file: ' + myFile + ' ' + str(j) + ' out of ' + str(files))
		
		fileRows = openFile(myFile, myAnnotatedFile, myRawFile)
		myRows += fileRows

		j = j + 1

		if j == maxNumFiles:
			break

	open(os.path.join(data_dir, output), 'w').close()
	writeToTSV(myRows, os.path.join(data_dir, output))

#opens both versions of the file, makes sure they're both okay
def openFile(coreNLPFileName, annotatedFileName, raw_file):

	rows = []
	flagNoLabels = False
	annotated_text = ''

	if annotatedFileName != None:
		try:
			parc_xml = open(annotatedFileName).read()
			corenlp_xml = open(coreNLPFileName).read()
			raw_text = open(raw_file).read()

			article = P(corenlp_xml, parc_xml, raw_text)

			filename = coreNLPFileName.split('/')[-1]
			#find the verbs in this file
			#extract the features from each verb
			#listOfVerbs = findVerbs(annotated_text, filename)
			listOfVerbs = findVerbs(article, filename)
			rows = prepareVerb(listOfVerbs, article, flagNoLabels)

		except:
			print 'error opening file'
			raise
			return rows


	else:
		corenlp_xml = open(coreNLPFileName).read()
		raw_text = open(raw_file).read()

		filename = coreNLPFileName.split('/')[-1]


		flagNoLabels = True
		parc_xml = None
		article = P(corenlp_xml, parc_xml, raw_text)

		listOfVerbs = findVerbs(article, filename)
		rows = prepareVerb(listOfVerbs, article, flagNoLabels)
		
	return rows, article

#use the constituency parse to find the verbs
def findVerbs(document,filename):
	verbPhraseList = []
		#skip over ROOT to beginning of the sentence S

		
		

	#implement cynthia's algorithm to find head verbs
	#procedureGETHEADVERBS(document) 
			#for VP in document do
				#if not VP has another VP as direct child then
					#for all children of VP do
						#if child is terminal node and child.PoS starts with VB then
							#add child to head verbs
	allVerbTokens = []
	for sentence in document.sentences:
		for token in sentence['tokens']:
			if token['pos'].startswith('V'):

				parent = token['c_parent']
				verbPhraseDependence = False
				children = parent['c_children']
				for child in children:
					if child['c_tag'] == 'VP':
						verbPhraseDependence = True
						continue

				if verbPhraseDependence:
					continue
				for child in children:
					if child['c_tag'].startswith('V') and child['word'] != None:
						allVerbTokens.append(child)



	#extract syntactic features (depth, parentNode, parentSiblingNodes)
	finalListVerbPhrases = []
	for verb in allVerbTokens:
		depth = verb['c_depth']
		parentNode = verb['c_parent']['c_tag']
		grandparents = verb['c_parent']['c_parent']
		if grandparents == None:
			continue
		auntsAndUncles = grandparents['c_children']
		parentSiblingNodes = []
		for aunt in auntsAndUncles:
			parentSiblingNodes.append(aunt['c_tag'])
		finalListVerbPhrases.append((verb['word'], verb['sentence_id'], verb['id'], filename, (depth,parentNode,parentSiblingNodes)))

	return finalListVerbPhrases


#takes verb and extracts all features
def prepareVerb(listOfVerbs, article, flagNoLabels):

	rows = []
	openQuote = False

	for sentence in article.sentences:
		sentenceID = sentence['id']

		beginnings = []
		endings = []

		for (word, sentID, tokenID, filename, syntacticFeatures) in listOfVerbs:
			if sentID == sentenceID:
				token = sentence['tokens'][tokenID]
				try:
					rowOfFeats = extractFeatures(token, sentence, filename, syntacticFeatures, openQuote)
					if rowOfFeats != None:
						rows.append(rowOfFeats)
				except:
					raise
		for token in sentence['tokens']:
			if (token['word'] == "''" or token['word'] == '``') and openQuote == False:
				openQuote = True
			elif (token['word'] == "''" or token['word'] == '``') and openQuote == True:
				openQuote = False

	return rows

#finds and assigns all the features 
def extractFeatures(token, sentence, filename, syntacticFeatures, openQuote):
	rowOfFeats = []

	verb = token['word']
	idVerb = token['id']

	Features = Verb(token['word'], token['lemma'], token['pos'])
	Features.set_metadata(sentence['id'], idVerb, filename)

	if token.has_key('attribution'):
		role = token['role']
		if role == 'cue':
			Features.set_label('Y')
		elif role == 'content':
			return None
		else:
			Features.set_label('N')
	else:
		Features.set_label('N')

	if idVerb > 0:
		prevToken = sentence['tokens'][idVerb - 1]
	else:
		prevToken = None
	if idVerb < len(sentence['tokens']) - 1:
		nexToken = sentence['tokens'][idVerb + 1]
	else:
		nexToken = None

	if prevToken != None:
		Features.set_previousToken(prevToken['word'], prevToken['lemma'], prevToken['pos'])

		if prevToken['word'] == ':':
			Features.set_colonAdjacent()
		elif prevToken['word'] == '``' or prevToken['word'] == "''":
			Features.set_quoteAdjacentInside()
	else:
		Features.set_previousToken('NONE!!', 'NONE!!', 'NONE!!')


	if nexToken != None:
		Features.set_nextToken(nexToken['word'], nexToken['lemma'], nexToken['pos'])

		if nexToken['word'] == ':':
			Features.set_colonAdjacent()
		elif nexToken['word'] == '``' or nexToken['word'] == "''":
			Features.set_quoteAdjacentInside()
	else:
		Features.set_nextToken('NONE!!', 'NONE!!', 'NONE!!')



	Features.set_verbNet(";!".join(vn.classids(token['lemma'])))
	Features.set_distances(token['id'], len(sentence['tokens']) - (token['id'] + 1))

	quoteMarkers = findQuoteMarkers(sentence, openQuote)
	FEATinQuotes = 'False'

	for (beg, end) in quoteMarkers:
		if idVerb > beg and idVerb < end:
			Features.set_insideQuotes()

	(depth, parentNode, parentSiblings) = syntacticFeatures
	Features.set_syntactic(depth, parentNode, ";!".join(parentSiblings))

	Features.makeList()
	rowOfFeats = Features.getList()
	
	return rowOfFeats

#identifies quote markers to see if there are mismatched quotes and returns 
#the index positions of each quotation mark
def findQuoteMarkers(sentence, openQuotes):
	begQuote = 0
	endQuote = 0
	listQuoteBeginnings = []
	listQuoteEndings = []
	found = False

	if openQuotes:
		listQuoteBeginnings = [-1]

	for quoteToken in sentence['tokens']:
		if quoteToken['word'] == '``' or (quoteToken['word'] == "''" and openQuotes == False):
			openQuotes = True
			listQuoteBeginnings.append(quoteToken['id'])
			found = True
		elif quoteToken['word'] == "''" and openQuotes == True:
			openQuotes = False
			listQuoteEndings.append(quoteToken['id'])
			found = True

	if found == False and openQuotes == True:
		return [(-1,len(sentence['tokens']))]



	if len(listQuoteBeginnings) > len(listQuoteEndings):
		listQuoteEndings.append(len(sentence['tokens']))
	elif len(listQuoteBeginnings) < len(listQuoteEndings):
		listQuoteBeginnings = [-1] + listQuoteBeginnings






	quoteMarkers = zip(listQuoteBeginnings, listQuoteEndings)

	return quoteMarkers

#parse command line args
def main():
	usageMessage = '\nCorrect usage of the Verb Cue Feature Extractor command is as follows: \n' + \
					'\n\n WHEN AN ANNOTATED FILESET EXISTS TO GET LABELS FROM:\n' + \
					'To extract verbs and their features: \n python source/intermediaries/verbCuesFeatureExtractor.py -labelled /pathToCoreNLPDirectory /pathToAnnotatedFilesDirectory /pathToRawFiles nameOfOutputFile.csv \n' + \
					'\nTo use the default path names for the PARC training data, and filename PARCTrainVerbFeats.csv please use the command with the label -default, as follows: \n' + \
					'\t python source/intermediaries/verbCuesFeatureExtractor.py -labelled -default' + \
					'\n\n WHEN THE LABELS ARE UNKNOWN:\n' + \
					'To extract verbs and their features: \n python source/intermediaries/verbCuesFeatureExtractor.py -unlabelled /pathToCoreNLPDirectory /pathToRaw nameOfOutputFile.csv \n' + \
					'\nFor reference, the path to the CoreNLP file is: /home/ndg/dataset/ptb2-corenlp/CoreNLP_tokenized/ + train, test or dev depending on your needs. \n' + \
					'The path to the Parc3 files is /home/ndg/dataset/parc3/ + train, test or dev depending on your needs.\n' + \
					'The path to the raw files is /home/ndg/dataset/ptb2-corenlp/masked_raw/ + train, test, or dev'

	args = sys.argv

	if len(args) == 6:

		flag = args[1]
		pathToCORENLP = args[2]
		pathToAnnotatedFiles = args[3] 
		pathToRaw = args[4]
		nameCSVOutput = args[5]


		if flag != '-labelled':
			print usageMessage
			return

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

		if os.path.isfile(data_dir + nameCSVOutput):
			print "That file already exists, you probably don't want to overwrite it"
			var = raw_input("Are you sure you want to overwrite this file? Please answer Y or N\n")
			if var == 'Y' or var == 'y':
				coreNLPFiles = openDirectory(pathToCORENLP)
				annotatedFiles = openDirectory(pathToAnnotatedFiles)
				rawFiles = openDirectory(pathToRaw)


				findFiles(coreNLPFiles, annotatedFiles, rawFiles, nameCSVOutput)
				return
			else:
				return

		else:
			print 'valid filename'

		coreNLPFiles = openDirectory(pathToCORENLP)
		annotatedFiles = openDirectory(pathToAnnotatedFiles)
		rawFiles = openDirectory(pathToRaw)


		findFiles(coreNLPFiles, annotatedFiles, rawFiles, nameCSVOutput)

	elif len(args) == 5:

		pathToCORENLP = args[2]
		nameCSVOutput = args[3]
		pathToRaw = args[4]

		if args[1] != '-unlabelled':
			print usageMessage
			return

		if os.path.isdir(pathToCORENLP):
			print 'valid path to a directory'
		else:
			print 'ERROR: The path to this coreNLP directory does not exist.'
			print usageMessage
			return

		if os.path.isfile(data_dir + nameCSVOutput):
			print "That file already exists, you probably don't want to overwrite it"
			var = raw_input("Are you sure you want to overwrite this file? Please answer Y or N\n")
			if var == 'Y' or var == 'y':
				coreNLPFiles = openDirectory(pathToCORENLP)
				rawFiles = openDirectory(pathToRaw)

				findFiles(coreNLPFiles, None, rawFiles, nameCSVOutput)
				return
			else:
				return

		coreNLPFiles = openDirectory(pathToCORENLP)
		rawFiles = openDirectory(pathToRaw)

		findFiles(coreNLPFiles, None, rawFiles, nameCSVOutput)

	elif len(args) == 3:
		if args[1] == '-labelled' and args[2] == '-default':
			pathToCORENLP = '/home/ndg/dataset/ptb2-corenlp/CoreNLP/train/'
			pathToAnnotatedFiles = '/home/ndg/dataset/parc3/train/'
			pathToRaw = '/home/ndg/dataset/ptb2-corenlp/masked_raw/train/'

			nameCSVOutput = 'PARCTrainFeatsAll.csv'

			coreNLPFiles = openDirectory(pathToCORENLP)
			annotatedFiles = openDirectory(pathToAnnotatedFiles)
			rawFiles = openDirectory(pathToRaw)


			findFiles(coreNLPFiles, annotatedFiles, rawFiles, nameCSVOutput)

		else:
			print usageMessage

	else:
		print usageMessage


#object Verb
#one attribute per feature
class Verb(object):

	FEATcolonAdjacency = 'False'
	FEATquotationAdjacency = 'False'
	FEATpreviousToken = None
	FEATpreviousLemma = None
	FEATpreviousPOS = None
	FEATinQuotes = 'False'
	FEATthisToken = None
	FEATthisLemma = None
	FEATthisPOS = None
	FEATnextLemma = None
	FEATnextToken = None
	FEATnextPOS = None
	FEATverbNetClasses = ''
	FEATdepth = None
	FEATparentNode = None
	FEATparentSiblings = ''
	FEATdistanceStart = None
	FEATdistanceEnd = None
	metadataSentId = None
	metadataTokenId = None
	metadataFilename = None
	label = None

	rowOfFeats = []

	def __init__(self, Token, Lemma, POS):
		self.FEATthisToken = Token
		self.FEATthisLemma = Lemma
		self.FEATthisPOS = POS	

	def set_colonAdjacent(self):
		self.FEATcolonAdjacency = 'True'

	def set_quoteAdjacentInside(self):
		self.FEATquotationAdjacency = 'True'


	def set_insideQuotes(self):
		self.FEATinQuotes = 'True'

	def set_previousToken(self, prevToken, prevLemma, prevPOS):
		self.FEATpreviousToken = prevToken
		self.FEATpreviousLemma = prevLemma
		self.FEATpreviousPOS = prevPOS

	def set_nextToken(self, nexToken, nexLemma, nextPOS):
		self.FEATnextToken = nexToken
		self.FEATnextLemma = nexLemma
		self.FEATnextPOS = nextPOS

	def set_verbNet(self, classes):
		self.FEATverbNetClasses = classes

	def set_syntactic(self, depth, parentNode, parentSiblings):
		self.FEATdepth = str(depth)
		self.FEATparentNode = parentNode
		self.FEATparentSiblings = parentSiblings

	def set_distances(self, start, end):
		self.FEATdistanceStart = str(start)
		self.FEATdistanceEnd = str(end)

	def set_metadata(self, sentID, tokID, filename):
		self.metadataSentId = str(sentID)
		self.metadataTokenId = str(tokID)
		self.metadataFilename = filename

	def set_label(self, value):
		self.label = value

	def makeList(self):
		self.rowOfFeats = ['thisToken=' + str(self.FEATthisToken), 'thisLemma=' + str(self.FEATthisLemma), 'thisPos=' + str(self.FEATthisPOS), \
						'lastToken=' + str(self.FEATpreviousToken), 'lastLemma=' + str(self.FEATpreviousLemma), 'lastPos=' + str(self.FEATpreviousPOS), \
						'nextToken=' + str(self.FEATnextToken), 'nextLemma=' + str(self.FEATnextLemma), 'nextPos=' + self.FEATnextPOS,\
						'colonAdj=' + self.FEATcolonAdjacency, 'quoteAdj=' + self.FEATquotationAdjacency, \
						'VNclasses='+str(self.FEATverbNetClasses), \
						'depth=' + self.FEATdepth, 'parentNode='+str(self.FEATparentNode), 'siblings='+str(self.FEATparentSiblings), \
						'distStart=' + self.FEATdistanceStart, 'distEnd='+self.FEATdistanceEnd, 'inQuotes=' + self.FEATinQuotes, \
						'label=' + self.label, 
						'metaData='+ self.metadataSentId + ';' + self.metadataTokenId + ';' + self.metadataFilename]


	def getList(self):
		return self.rowOfFeats


class Stack:
     def __init__(self):
         self.items = []

     def isEmpty(self):
         return self.items == []

     def push(self, item):
         self.items.append(item)

     def pop(self):
         return self.items.pop()

     def peek(self):
         return self.items[len(self.items)-1]

     def size(self):
         return len(self.items)



if __name__ == '__main__':
   main()