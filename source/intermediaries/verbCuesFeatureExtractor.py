print 'importing'
from nltk.tree import *
from nltk.corpus import verbnet as vn
import csv
import os
import sys
from parc_reader import ParcCorenlpReader as P
from intermediaries.nlpReaders.annotated_text import AnnotatedText as A
from intermediaries.nlpReaders.parc_reader import AnnotatedText as B
import reformatLabelledVerbs as reformat
print 'imports done'

maxNumFiles = -1
minNumFile = 0
data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'data/'))

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

#get the files, open them, extract verbs and features and create a large array of rows
def findFiles(listOfNLPFiles, listOfAnnotatedFiles, listOfRawFiles, output):

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

			#annotated_text = A(corenlp_xml)
			article = P(corenlp_xml, parc_xml, raw_text)

			filename = coreNLPFileName.split('/')[-1]
			#find the verbs in this file
			#extract the features from each verb
			listOfVerbs = findVerbs(article, filename)
			rows = prepareVerb(listOfVerbs, article, flagNoLabels)

		except:
			print 'error opening file'
			raise
			return rows


	else:
		corenlp_xml = open(coreNLPFileName).read()
		raw_text = open(raw_file).read()

		#annotated_text = A(corenlp_xml)
		filename = coreNLPFileName.split('/')[-1]
		article = P(corenlp_xml, parc_xml, raw_text)

		flagNoLabels = True
		parc_xml = None
		listOfVerbs = findVerbs(article, filename)

		rows = prepareVerb(listOfVerbs, article, flagNoLabels)
		
	return rows, article

#use the constituency parse to find the verbs
def findVerbs(document, filename):


	for sentence in document.sentences:
		print sentence
		print sentence.keys()





	listOfParse = []

	sent_tags = document.soup.find('sentences').find_all('sentence')
	for s in sent_tags:
		listOfParse.append(s.find('parse').text)
	
	listOfVPs = []
	sentenceID = 0
	#create an NLTK tree from each parse of the sentence
	for parse in listOfParse:
		sentence = document.sentences[sentenceID]
		for item in sentence['tokens']:
			pos = item['pos']
			if '(' in item['word']:
				parse = parse.replace('(' + pos + ' ()', '(' + pos + ' -LRB-)')
			elif ')' in item['word']:
				parse = parse.replace('(' + pos + ' ))', '(' + pos + ' -RRB-)')

		tree = ParentedTree.fromstring(parse)

		#treepositions returns a list that looks like 
		#[(), (0,), (0, 0), (0, 0, 0), (0, 1), (0, 1, 0), (1,), (1, 0), (1, 0, 0), ...]
		#which documents each internal node/leaf's position in the tree
		positions = tree.treepositions()
		next = None
		posCount = 0
		listPositions = []

		#for each position in the tree's position, if the length of the elemnt is greater than
		#the previous element's, then we've found a leaf in the subtree

		#this documents those leaves, and provides an index that matches the token ID in the 
		#core NLP data

		#this is necessary because the Parse Trees don't have token based ids, so we face issues
		#when two tokens are the same as it can not resolve which is which.

		#these tree positions help us to identify the exact position of the leaf in the tree

		for indx, pos in enumerate(positions):
			if indx == len(positions)-1:
				break
			listPositions.append((pos, posCount))
			next = positions[indx + 1]
			if len(positions[indx]) > len(next):
				posCount += 1


		#find all subtrees that match the label Verb Phrase
		#implement Pareti's algorithm 

		#procedureGETHEADVERBS(document) 
			#for VP in document do
				#if not VP has another VP as direct child then
					#for all children of VP do
						#if child is terminal node and child.PoS starts with VB then
							#add child to head verbs
		for subtree in tree.subtrees(filter = lambda t: t.label() == 'VP'):
			childArray = []
			noVPChild = True

			for child in subtree:
				childArray.append(child)
				if child.label() == 'VP':
					noVPChild = False

			if noVPChild:
				for child in childArray:
					if len(child) == 1 and child.label().startswith('VB'):
						for (position, idTok) in listPositions:
							if position == child.treeposition():
								tokenID = idTok
								depth = len(child.treeposition())
								#gather all tree related data
								syntacticFeatures = extractSyntactic(tree, child)

						#append all this to the verb phrase list
						verbTuple = child[0], tokenID, sentenceID, filename, syntacticFeatures
						listOfVPs.append(verbTuple)
		sentenceID += 1
	
	return listOfVPs

def extractSyntactic(tree, child):
	#tree.pretty_print()
	#child.pretty_print()

	#the length of the treeposition tells us how many intermediary nodes
	#had to be traversed, e.g. the depth
	depth = len(child.treeposition())

	parent = child.parent()
	parentNode = parent.label()

	leftLabels = []

	found = False
	thisparent = child.parent()
	#finds all left siblings
	while found == False:
		leftSibling = thisparent.left_sibling()
		if leftSibling == None:
			found = True
		else:
			thisparent = leftSibling
			leftLabels.append(leftSibling.label())

	rightLabels = []

	found = False
	thisparent = child.parent()
	#finds all right siblings
	while found == False:
		rightSibling = thisparent.right_sibling()
		if rightSibling == None:
			found = True
		else:
			thisparent = rightSibling
			rightLabels.append(rightSibling.label())

	parentSiblings = leftLabels + rightLabels

	return (depth, parentNode, parentSiblings)

def prepareVerb(listOfVerbs, article, flagNoLabels):

	rows = []


	for sentence in article.sentences:
		sentenceID = sentence['id']

		for (word, tokenID, sentID, filename, syntacticFeatures) in listOfVerbs:
			if sentID == sentenceID:
				token = sentence['tokens'][tokenID]
				try:
					rowOfFeats = extractFeatures(token, sentence, filename, syntacticFeatures)
					rows.append(rowOfFeats)
				except:
					raise

	return rows

#finds and assigns all the features 
def extractFeatures(token, sentence, filename, syntacticFeatures):
	rowOfFeats = []

	verb = token['word']
	idVerb = token['id']

	Features = Verb(token['word'], token['lemma'], token['pos'])
	Features.set_metadata(sentence['id'], idVerb, filename)

	if token.has_key('attribution'):
		role = token['role']
		if role == 'cue':
			Features.set_label('Y')
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
		elif prevToken['word'] == '``':
			Features.set_quoteAdjacentInside()
		elif prevToken['word'] == "''":
			Features.set_quoteAdjacentOutside()
		elif prevToken['word'] == ',':
			beforeComma = sentence['tokens'][idVerb - 2]
			if beforeComma['word'] == '``':
				Features.set_quoteAdjacentInside()
			elif beforeComma['word'] == "''":
				Features.set_quoteAdjacentOutside()


	if nexToken != None:
		Features.set_nextToken(nexToken['word'], nexToken['lemma'], nexToken['pos'])

		if nexToken['word'] == ':':
			Features.set_colonAdjacent()
		elif nexToken['word'] == '``':
			Features.set_quoteAdjacentOutside()
		elif nexToken['word'] == "''":
			Features.set_quoteAdjacentInside()
		elif nexToken['word'] == ',':
			try:
				afterComma = sentence['tokens'][idVerb + 2]
				if afterComma['word'] == '``':
					Features.set_quoteAdjacentOutside()
				elif afterComma['word'] == "''":
					Features.set_quoteAdjacentInside()
			except:
				print 'out of range'
	else:
		Features.set_nextToken('NONE!!', 'NONE!!', 'NONE!!')



	Features.set_verbNet(";!".join(vn.classids(token['lemma'])))
	Features.set_distances(token['id'], len(sentence['tokens']) - (token['id'] + 1))

	quoteMarkers = findQuoteMarkers(sentence)
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
def findQuoteMarkers(sentence):
	begQuote = 0
	endQuote = 0
	listQuoteBeginnings = []
	listQuoteEndings = []

	for quoteToken in sentence['tokens']:
		if quoteToken['word'] == '``':
			openQuotes = True
			listQuoteBeginnings.append(quoteToken['id'])
		elif quoteToken['word'] == "''":
			openQuotes = False
			listQuoteEndings.append(quoteToken['id'])


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
		nameCSVOutput = args[4]
		pathToRaw = args[5]

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
			pathToCORENLP = '/home/ndg/dataset/ptb2-corenlp/CoreNLP_tokenized/train/'
			pathToAnnotatedFiles = '/home/ndg/dataset/parc3/train/'
			pathToRaw = '/home/ndg/dataset/ptb2-corenlp/masked_raw/train/'

			nameCSVOutput = 'PARCTrainVerbFeats.csv'

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
	FEATquotationAdjacencyInside = 'False'
	FEATquotationAdjacencyOutside = 'False'
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
		self.FEATquotationAdjacencyInside = 'True'

	def set_quoteAdjacentOutside(self):
		self.FEATquotationAdjacencyOutside = 'True'

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
						'colonAdj=' + self.FEATcolonAdjacency, 'quoteAdjIns=' + self.FEATquotationAdjacencyInside,'quoteAdjOut=' + self.FEATquotationAdjacencyOutside, \
						'VNclasses='+str(self.FEATverbNetClasses), \
						'depth=' + self.FEATdepth, 'parentNode='+str(self.FEATparentNode), 'siblings='+str(self.FEATparentSiblings), \
						'distStart=' + self.FEATdistanceStart, 'distEnd='+self.FEATdistanceEnd, 'inQuotes=' + self.FEATinQuotes, \
						'label=' + self.label, 
						'metaData='+ self.metadataSentId + ';' + self.metadataTokenId + ';' + self.metadataFilename]


	def getList(self):
		return self.rowOfFeats


if __name__ == '__main__':
   main()