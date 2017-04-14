import sys
import os
from corenlp_xml_reader import AnnotatedText as A
#from intermediaries.nlpReaders.annotated_text import AnnotatedText as A
from intermediaries.nlpReaders.parc_reader import AnnotatedText as B
from parc_reader import ParcCorenlpReader as P
from nltk.tree import *
import csv
import multiprocessing
from multiprocessing import Manager

#fixes all the omnipresent unicode issues
print sys.getdefaultencoding()
reload(sys)
sys.setdefaultencoding('utf-8')

#change this if you would only like to do a certain number of files, useful for testing
maxNumFiles = 1000

#base dir for all data files
data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..', 'data/'))

#store the WordNet generated lists of hyponyms
with open(data_dir + '/peopleHyponyms.csv', 'rb') as f:
    reader = csv.reader(f)
    peopleHyponyms = list(reader)

with open(data_dir + '/orgHyponyms.csv', 'rb') as f:
	reader = csv.reader(f)
	orgHyponyms = list(reader)

with open(data_dir + '/nounCues.csv', 'rb') as f:
    reader = csv.reader(f)
    nounCues = list(reader)

peopleHyponyms = peopleHyponyms[0]
orgHyponyms = orgHyponyms[0]
nounCues = nounCues[0]


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

#open verb file, extract what we need and create a verbs list
def openVerbCues(verbCuesFile):
	verbsList = []
	with open(os.path.join(data_dir, verbCuesFile), 'rb') as f:
		reader = csv.reader(f)
		verbsList = list(reader)
	newVerbsList = []

	for indx, verb in enumerate(verbsList):
		metadata = verb[1].split(';')
		sentID =  metadata[0]
		tokID = metadata[1]
		fileName = metadata[2]
		newVerbsList.append([verb[0], sentID, tokID, fileName, verb[2]])

	return newVerbsList

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in xrange(0, len(l), n))

def workerFunction(myFile, listOfAnnotatedFiles, listOfRawFiles, output, verbCuesFile, return_list):

	flagNoLabels = False

	files = len(listOfNLPFiles)
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

	print('opening file: ' + myFile + ' ' + str(j) + ' out of ' + str(files))
	
	#extract the verbs whose metadata filename matches this NLP file
	specificFileVerbs = []
	for verb in verbList:
		if (verb[3] == filename):
			specificFileVerbs.append(verb)

	#open the file, extract the features and return all the rows
	fileRows = openFile(myFile, myAnnotatedFile, myRawFile, specificFileVerbs)

	return_list += fileRows




#get the files, open them, extract verbs and features and create a large array of rows
def findFiles(listOfNLPFiles, listOfAnnotatedFiles, listOfRawFiles, output, verbCuesFile):

	verbList = openVerbCues(verbCuesFile)

	splitLists = list(chunks(coreNlPFiles, len(coreNlPFiles)/10))
	j = 0

	lastList = splitLists[-1]
	del splitLists[-1]

	lengthLists = len(splitLists[0])

	jobs = []
	manager = Manager()
	return_list = manager.list()


	if listOfAnnotatedFiles == None:
		flagNoLabels = True
	else:
		flagNoLabels = False

	#first lists are all equally sized, pick one from each at each iteration
	for i in range(lengthLists):
		if i == -1:
			break
		for thisList in splitLists:
			myFile = thisList[i]
			p = multiprocessing.Process(target = workerFunction, args=(myFile, listOfAnnotatedFiles, listOfRawFiles, output, verbCuesFile, return_list))
			jobs.append(p)
			p.start()

	

	#append the files from last list (remainder of total files divided by 10)
	for myFile in lastList:
		p = multiprocessing.Process(target = workerFunction, args=(myFile, listOfAnnotatedFiles, listOfRawFiles, output, verbCuesFile, return_list))
		jobs.append(p)
		p.start()

	for proc in jobs:
		proc.join()

	open(os.path.join(data_dir, output), 'w').close()
	writeToTXT(return_list, os.path.join(data_dir, output), flagNoLabels)

def openFile(coreNLPFileName, annotatedFileNamnscccccce, raw_file, verbList):

	allrows = []
	#open annotated if it exists
	if annotatedFileName != None:
		try:
			parc_xml = open(annotatedFileName).read()
			corenlp_xml = open(coreNLPFileName).read()
			raw_text = open(raw_file).read()
			annotated_text = A(corenlp_xml)
			article = P(corenlp_xml, parc_xml, raw_text)


		except:
			print 'error opening file'
			return rows

	else:
		parc_xml = None
		corenlp_xml = open(coreNLPFileName).read()
		raw_text = open(raw_file).read()
		annotated_text = A(corenlp_xml)
		article = P(corenlp_xml, parc_xml, raw_text)

	filename = coreNLPFileName.split('/')[-1]

	rows = findFeatures(filename, article, annotated_text, verbList, annotatedFileName)
	allrows += rows

	return rows, article


def writeToTXT(rows, filename, flagNoLabels):
	#if the data is unlabelled, we create a second metadata file that stores the word
	#as well as the filename and sentence ID
	#this file will be used to reconstitute the spans once CRFsuite gets through them
	if flagNoLabels == True:
		newRows = []
		metadataRows = []
		token = ''
		for row in rows:
			row = row.split('\t')
			metadata = row[-1]

			for column in row:
				if 'word[0]=' in column and 'word[-1]|word[0]=' not in column:
					token = column

			metadataRows.append(metadata + '\t' + token)
			del row[-1]
			row = '\t'.join(row)
			newRows.append(row)

		rows = newRows

		#make a new filename with METADATA in it
		fileRow = filename.split('/')
		thisfile = fileRow[-1]
		del fileRow[-1]
		thisfile = 'METADATA' + thisfile
		fileRow.append(thisfile)
		metafile = '/'.join(fileRow)

		with open(metafile, 'w') as myfile:
			for row in metadataRows:
				myfile.write(row + '\n')
		myfile.close()

	else:
		newRows = []

		for row in rows:
			row = row.split('\t')
			del row[-1]

			row = '\t'.join(row)
			newRows.append(row)

		rows = newRows

	#write all the tokens and their features to a txt
	with open(filename, 'w') as myfile:
		for row in rows:
			myfile.write(row + '\n')
	myfile.close()

	print '\nData written to ' + filename + '\n'

#gather all the features and create rows
def findFeatures(filename, article, corenlp_xml, verbsList, annotatedFileName):
	print filename		

	rows = []

	#find the constituency parse of the sentences
	listOfParse = []
	sent_tags = corenlp_xml.soup.find('sentences').find_all('sentence')
	for s in sent_tags:
		listOfParse.append(s.find('parse').text)
	i = 0

	print('extracting features ......')
	openQuotes = 'false'

	lastLabel = 'O'

	#begin extracting features
	for sentence in article.sentences:
		
		lengthOfSentence = len(sentence['tokens'])
		idOfSentence = sentence['id']

		currentSentPers = 'false'
		
		beginningQuote = 'false'
		parseTree = listOfParse[i]
		i = i + 1

		#these are the features that will stay the same across each token in the sentence
		#e.g. containsOrganization, containsNamedEntity etc.
		#returns the array for this sentence of tokens, pos, lemma which will be used to
		#constitute the token based features
		rowSentFeats = ''
		tokenArray, posArray, lemmaArray,  peopleHyponym, orgHyponym, rowSentFeats = \
						getSentenceFeatures(sentence, rowSentFeats)

		#get verb cue, sentence wide features, e.g. containsVerbCue, verbCueNearTheEnd?
		rowSentFeats = getVerbListSentenceFeatures(sentence, verbsList, rowSentFeats)

		#rowSentFeats now contains a string that looks like
		#'containsOrg='True'\tcontainsVerbCue='True'\t.....'
		prevSyntactic = ''
		for token in sentence['tokens']:

			row = rowSentFeats

			word = str(token['word'])
			pos = str(token['pos'])
			lemma = str(token['lemma'])
			idOfToken = token['id']



			if annotatedFileName != None:
				#assign labels
				label = token['attribution']
				role = token['role']
	

				if label == None:
					row = 'O\t' + row
					lastLabel = 'O'
				elif role == 'content' and lastLabel == 'O':
					row = 'B\t' + row
					lastLabel = 'B'
				elif role == 'content' and lastLabel == 'B':
					row = 'I\t' + row
					lastLabel = 'I'
				elif role == 'content':
					row = 'I\t' + row
					lastLabel = 'I'
				else:
					row = 'O\t' + row
					lastLabel = 'O'
			else:
				row = '\t' + row
			

			if "''" in str(token['word']):
				openQuotes = 'false'

			#append the features
			row += 'sentenceLength=' + str(lengthOfSentence) + '\t'


	
			row = getTokenFeatures(idOfToken, tokenArray, row, 'word')
			row = getTokenFeatures(idOfToken, posArray, row, 'pos')
			row = getTokenFeatures(idOfToken, lemmaArray, row, 'lemma')

			row = getTokenPairs(idOfToken, tokenArray, row, 'word')
			row = getTokenPairs(idOfToken, posArray, row, 'pos')
			row = getTokenPairs(idOfToken, lemmaArray, row, 'lemma')

			row = findRelations(token, row)

			for (idHypo, hyponym) in peopleHyponym:
				row += 'personHyponym[' + str(idHypo - idOfToken) + ']=' + hyponym + '\t'

			for (idHypo, hyponym) in orgHyponym:
				row += 'organizationHyponym[' + str(idHypo - idOfToken) + ']=' + hyponym + '\t'

			#the inside quote labels
			if openQuotes == 'true' and beginningQuote == 'true':
				row += "insidequotes='true'\t"


			if "``" in str(token['word']):
				openQuotes = 'true'
				beginningQuote = 'true'

			row = getConstituentLabels(parseTree, token, sentence, row)

			prevSyntactic, row = getSyntactic(parseTree, token, sentence, row, verbsList, prevSyntactic)

			row += 'filename=' + filename + '\tsentenceID=' + str(idOfSentence) + '\ttokenID=' + str(idOfToken)
			rows.append(row)


	return rows

#get verbCue sentence wide features
def getVerbListSentenceFeatures(sentence, verbs, rowSentFeats):
	containsVerbCue = False
	verbCueNearEnd = False
	sentenceID = sentence['id']

	sentenceLen = len(sentence['tokens'])
	nearEnd = sentenceLen - min(4, sentenceLen/2)

	for verb in verbs:
		if str(verb[1]) == str(sentenceID):
			sentVerb = sentence['tokens'][int(verb[2])]['word']

			if sentVerb == verb[0]:
				if verb[4] == 'Y':
					containsVerbCue = True
					rowSentFeats += '''containsVerbCue='true'\t'''
					if int(verb[2]) >= nearEnd:
						rowSentFeats += '''verbCueNearEnd='true'\t'''


	return rowSentFeats


#identifies which relations the token has with its parents and children
#appends seperate features for the relation as well as the relation with the word itself
def findRelations(token, row):
	listRelations = ['acl', 'acl:relcl', 'advcl', 'advmod', 'amod', 'appos', 'aux', 'auxpass', 'case', 'cc', 'cc:preconj', 
								'ccomp', 'compound', 'compound:prt', 'conj', 'cop', 'csubj', 'csubjpass', 'dep', 'det', 'det:predet',
								'discourse', 'dislocated', 'dobj', 'expl', 'foreign', 'goeswith', 'iobj', 'list', 'mark', 'mwe', 'name',
								'neg', 'nmod', 'nmod:npmod', 'nmod:poss', 'nmod:tmod', 'nsubj', 'nsubjpass', 'nummod', 'parataxis', 
								'punct', 'remnant', 'reparandum', 'root', 'vocative', 'xcomp']




	if (token.has_key('parents')):
		for parents in token['parents']:
			relation, parent = parents
			if relation in listRelations:
				row += 'p-Relation=' + relation + '\t'
				row += 'p-Relation|p-token=' + relation + '|' + parent['word'] + '\t'

			else:
				rel = relation.split(':')
				row += 'p-Relation=' + rel[0] + '\t'
				row += 'p-Relation|p-token=' + rel[0] + '|' + parent['word'] + '\t'


	if (token.has_key('children')):
		for aChild in token['children']:
			relation, child = aChild
			if relation in listRelations:
				row += 'c-Relation=' + relation + '\t'
				row += 'c-Relation|c-token=' + relation + '|' + child['word'] + '\t'

			else:
				rel = relation.split(':')
				row += 'c-Relation=' + rel[0] + '\t'
				row += 'c-Relation|c-token=' + rel[0] + '|' + child['word'] + '\t'

	return row

#gets individual features, position indexed
#part of speech, lemma, word
def getTokenFeatures(idOfToken, array, row, name):
	bottomBound = -5

	if idOfToken < 5:
		bottomBound = 0 - idOfToken


	topBound = len(array) - idOfToken

	if topBound > 5:
		topBound = 6

	j = idOfToken

	while (bottomBound < topBound):
		row += name + '[' + str(bottomBound) + ']=' + array[j + bottomBound] + '\t'
		bottomBound = bottomBound + 1

	return row


#gets the pairs of features from -5 to 5, position indexed
#works for PartOfSpeech, Lemma, Word
def getTokenPairs(idOfToken, array, row, name):
	bottomBound = -5

	if idOfToken < 5:
		bottomBound = 0 - idOfToken

	topBound = len(array) - idOfToken

	if topBound > 5:
		topBound = 6


	j = idOfToken

	while (bottomBound < topBound - 1):
		row += name + '[' + str(bottomBound) + ']|' + name + '[' + str(bottomBound + 1) \
						+ ']=' + array[j + bottomBound] + '|' + array[j + bottomBound + 1] + '\t'
		bottomBound = bottomBound + 1

	return row


#gets a series of trues and falses depending on whether the sentence contains any 
#of the key features
def getSentenceFeatures(sentence, row):
	tokenArray = []
	posArray = []
	lemmaArray = []
	peopleHyponym = []
	orgHyponym = []

	foundPerson=False
	foundOrganization=False
	foundPronoun=False
	foundQuotes=False

	foundAccording = False
	possibleNounCue = False

	foundAny = False

	for token in sentence['tokens']:
			namedEnt = str(token['ner'])
			if (namedEnt == 'PERSON' and foundPerson==False):
				row += '''containsPerson='true'\t'''
				foundPerson = True
			elif (namedEnt == 'ORGANIZATION' and foundOrganization==False):
				row += '''containsOrganization='true'\t'''
				foundOrganization = True
			pos = str(token['pos'])
			if 'PRP' in pos and foundPronoun==False:
				row += '''containsPronoun='true'\t'''
				foundPronoun=True
			word = str(token['word'])
			if "''" in word and foundQuotes==False:
				row += '''containsQuotes='true'\t'''
				foundQuotes=True

			tokenArray = tokenArray + [word]
			posArray = posArray + [pos]
			lemmaArray = lemmaArray + [str(token['lemma'])]

			#get positions of possible hyponyms within sentence
			if word.lower() in peopleHyponyms:
				peopleHyponym.append((token['id'], word))

			if word.lower() in orgHyponyms:
				orgHyponym.append((token['id'], word))

			if (word.lower() != 'to' and foundAccording == True):
				foundAccording = False
			if (word.lower() == 'according'):
				foundAccording = True
			if (word.lower() == 'to' and foundAccording == True):
				row += '''containsAccordingTo='true'\t'''
				foundAccording = False

			if foundPerson == True or foundOrganization == True or foundPronoun == True or foundQuotes == True or foundAccording == True:
				row += '''foundAnySentenceFeatures='true'\t'''
				foundAny = True

			#if word.lower() in nounCues and '''containsNounCue='true''' not in row:
			#	row += '''containsNounCue='true'\t'''

	return tokenArray, posArray, lemmaArray, peopleHyponym, orgHyponym, row

#use the parse tree to set up finding the constituencies
def getConstituentLabels(parseTree, token, sentence, row):
	subTree = sentence['parse']

	listOfWords = []

	listofConstituences =  getPaths(subTree, listOfWords, token, sentence)

	if (listofConstituences != None):
		for (lab, dep) in listofConstituences:
			row += 'const=(' + lab + ',' + str(dep) + ')\t'

	return row	



#use a stack, go through each word 
def getPaths(treeDict, listOfWords, token, sentence):
	targetWord = str(token['word'])

	word = treeDict['word']
	s = Stack()
	s.push(treeDict)

	currIdentity = len(sentence['tokens'])
	while not (s.isEmpty()):
		currTreeDict = s.pop()
		thisWord = currTreeDict['word']

		if thisWord != None:
			currIdentity = currIdentity - 1

		#if we found the token's word
		if thisWord == targetWord and currIdentity == token['id']:
			OGdepth = currTreeDict['depth']
			parent = currTreeDict['parent']
			#finding each parent and their constituents
			if parent.has_key('depth') and parent['depth'] != 0:
				while (parent['depth'] != 1):
					code = str(parent['code'])
					depth = parent['depth'] - 1
					constTuple = (code, depth)
					parent = parent['parent']
					listOfWords.append(constTuple)
					if len(listOfWords) == OGdepth - 2:
						myList = listOfWords
						return myList
			else:
				return listOfWords
		#push all the children onto the stack 
		else:
			for child in currTreeDict['children']:
				s.push(child)

#use the parse tree to find all the syntactic info
def getSyntactic(parseTree, token, sentence, row, verbsList, prevSyntactic):



	targetWord = str(token['word'])
	idOfWord = token['id']

	for item in sentence['tokens']:
		pos = item['pos']
		if '(' in item['word']:
			parseTree = parseTree.replace('(' + pos + ' ()', '(' + pos + ' -LRB-)')
		elif ')' in item['word']:
			parseTree = parseTree.replace('(' + pos + ' ))', '(' + pos + ' -RRB-)')

	tree = ParentedTree.fromstring(parseTree)



	#reformat the text to match properly
	if targetWord == '(':
		targetWord = '-LRB-'
	if targetWord == ')':
		targetWord = '-RRB-'

	#get the indices for all the leaves that match this token
	indices = [i for i, x in enumerate(tree.leaves()) if x == targetWord]

	occurence = 0
	scopingId = 0

	if idOfWord != tree.leaves().index(targetWord):
		occurence = indices.index(idOfWord)
	scopingId = occurence

	#if there aren't multiple occurences of the same token
	if occurence == 0:
		gen = tree.subtrees(lambda tree2: str(tree2.leaves()[0]) == targetWord)
		try:
			subtree = gen.next()
		except:
			print sys.exc_info()[0]
			print 'error collecting subtree'
			return prevSyntactic, row
	#find correct token within the sentence
	else:
		next = 'false'
		for mytree in tree.subtrees(lambda tree2: str(tree2.leaves()[0]) == targetWord):

			if next == 'true' and occurence == 0:
				subtree = mytree
				break

			else:
				next = 'false'

			if mytree.height() == 2:
				next = 'true'
				occurence = occurence - 1

	#get subtree's label, length of span and depth
	flattened = subtree.flatten()
	label = flattened.label()
	lengthSpan = len(flattened.leaves())
	depth = len(subtree.treeposition())

	#find the ID in the original sentence so that we can find out whether any of the words are verb cues
	tokenArray = []
	idOfWord = None
	for token in sentence['tokens']:
		word = token['word']
		if word == '(':
			word = '-LRB-'
		if word == ')':
			word = '-RRB-'

		if targetWord == word and scopingId == 0:
			idOfWord = token['id']
			break
		elif targetWord == word:
			scopingId = scopingId - 1
		else:
			continue

	tokenArray = tree.leaves()


	#check if any tokens in the span are verb cues and append if found
	constHasSpan = False
	for verb in verbsList:
		verbWord = verb[0]
		verbSent = int(verb[1])
		verbTok = int(verb[2])
		verbLabel = verb[4]

		if verbSent != sentence['id']:
			continue
		else:
			for i in range(len(flattened.leaves())):
				if i + idOfWord == verbTok and tokenArray[i + idOfWord] == verbWord and verbLabel == 'Y':
					row += '''constSpanVC='true'\t'''
					constHasSpan = True


	row += 'constLabel=' + label + '\t' + 'constSpanLength=' + str(lengthSpan) + '\t' + 'depthSpan=' + str(depth) + '\t'

	#get the subtree's parent tree
	parentTreePosList = list(subtree.treeposition())
	#no parent, return the row as it is
	if len(parentTreePosList) == 0:
		return prevSyntactic, row
	elif targetWord == '-LRB-' or targetWord == 'RRB':
		return prevSyntactic, row
	
	parentTreePosList.pop()
	parentTreeHead = tuple(parentTreePosList)
	parentTree = tree[parentTreePosList]
	parentFlat = parentTree.flatten()
	parentLabel = parentFlat.label()
	lengthSpanParent = len(parentFlat.leaves())
	parentDepth = len(parentTree.treeposition())


	#find correct word id that begins the span
	begIndex = None
	for indx, word in enumerate(tokenArray):
		i = 0
		if parentFlat.leaves()[0] == word:

			for item in parentFlat.leaves():

				if i + indx == len(tokenArray):
					continue
				if item == tokenArray[i + indx] and i == len(parentFlat.leaves()) - 1:
					begIndex = indx
				if item == tokenArray[i + indx]:
					i = i + 1


	#find out if any of these are verbs

	for verb in verbsList:
			verbWord = verb[0]
			verbSent = int(verb[1])
			verbTok = int(verb[2])
			verbLabel = verb[4]

			if verbSent != sentence['id']:
				continue
			else:
				for i in range(len(parentFlat.leaves())):

						if i + begIndex == verbTok and tokenArray[i + begIndex] == verbWord and verbLabel == 'Y':
							row += '''parentConstSpanVC='true'\t'''
							#no need to add it twice
							constHasSpan = False



	#if the child tree had a verb cue, then we know the parent does 
	if constHasSpan == True:
		row += '''parentConstSpanVC='true'\t'''

	row += 'parentConstLabel=' + parentLabel + '\t' + 'parentConstSpanLength=' +  \
							str(lengthSpanParent) + '\t' + 'parentDepthSpan=' + str(parentDepth) + '\t'
					
	return prevSyntactic, row

#parse command line arguments
def main():
	usageMessage = '\nCorrect usage of the Content Span Extractor command is as follows: \n' + \
					'\n\n WHEN AN ANNOTATED FILESET EXISTS TO GET LABELS FROM:\n' + \
					'To extract tokens and their features: \n python source/intermediaries/contentSpanExtractor.py -labelled /pathToCoreNLPDirectory /pathToAnnotatedFilesDirectory /pathToRawDirectory nameCorrespondingTaggedVerbCuesFile nameOfOutputFile.txt \n' + \
					'\nTo use the default path names for the PARC training data, and filename PARCtrainContentSpans.txt please use the command with the label -default, as follows: \n' + \
					'\t python source/intermediaries/contentSpanExtractor.py -labelled -default' + \
					'\n\n WHEN THE LABELS ARE UNKNOWN:\n' + \
					'To extract tokens and their features: \n python source/intermediaries/contentSpanExtractor.py -unlabelled /pathToCoreNLPDirectory /pathToRawDirectory nameCorrespondingTaggedVerbCuesFile nameOfOutputFile.txt \n' + \
					'\nFor reference, the path to the CoreNLP file is: /home/ndg/dataset/ptb2-corenlp/CoreNLP/ + train, test or dev depending on your needs. \n' + \
					'The path to the Parc3 files is /home/ndg/dataset/parc3/ + train, test or dev depending on your needs.\n'

	args = sys.argv

	if len(args) == 7:

		flag = args[1]
		pathToCORENLP = args[2]
		pathToAnnotatedFiles = args[3] 
		pathToRaw = argsg[4]
		verbCuesFile = args[5]
		nameTxtOutput = args[6]

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

		if os.path.isfile(data_dir + nameTxtOutput):
			print "That file already exists, you probably don't want to overwrite it"
			var = raw_input("Are you sure you want to overwrite this file? Please answer Y or N\n")
			if var == 'Y' or var == 'y':

				coreNLPFiles = openDirectory(pathToCORENLP)
				annotatedFiles = openDirectory(pathToAnnotatedFiles)
				rawFiles = openDirectory(pathToRaw)

				findFiles(coreNLPFiles, annotatedFiles, rawFiles, nameTxtOutput, verbCuesFile)
				return
			else:
				return

		else:
			print 'valid filename'

		coreNLPFiles = openDirectory(pathToCORENLP)
		annotatedFiles = openDirectory(pathToAnnotatedFiles)
		rawFiles = openDirectory(pathToRaw)


		findFiles(coreNLPFiles, annotatedFiles, rawFiles, nameTxtOutput, verbCuesFile)

	elif len(args) == 6:

		pathToCORENLP = args[2]
		pathToRaw = args[3]
		verbCuesFile = args[4]
		nameTxtOutput = args[5]

		if args[1] != '-unlabelled':
			print usageMessage
			return

		if os.path.isdir(pathToCORENLP):
			print 'valid path to a directory'
		else:
			print 'ERROR: The path to this coreNLP directory does not exist.'
			print usageMessage
			return

		if os.path.isfile(data_dir + nameTxtOutput):
			print "That file already exists, you probably don't want to overwrite it"
			var = raw_input("Are you sure you want to overwrite this file? Please answer Y or N\n")
			if var == 'Y' or var == 'y':
				coreNLPFiles = openDirectory(pathToCORENLP)
				rawFiles = openDirectory(pathToRaw)

				findFiles(coreNLPFiles, None, rawFiles, nameTxtOutput, verbCuesFile)
				return
			else:
				return
		coreNLPFiles = openDirectory(pathToCORENLP)
		rawFiles = openDirectory(pathToRaw)

		findFiles(coreNLPFiles, None, rawFiles, nameTxtOutput, verbCuesFile)

	elif len(args) == 3:
		if args[1] == '-labelled' and args[2] == '-default':
			pathToCORENLP = '/home/ndg/dataset/ptb2-corenlp/CoreNLP_tokenized/train/'
			pathToAnnotatedFiles = '/home/ndg/dataset/parc3/train/'
			pathToRaw = '/home/ndg/dataset/ptb2-corenlp/masked_raw/train/'

			verbCuesFile = 'train/PARCTrainVerbFeatsFOR_SPAN_EXTRACTOR.csv'
			nameTxtOutput = 'PARCTrainContentSpans.txt'
			coreNLPFiles = openDirectory(pathToCORENLP)
			annotatedFiles = openDirectory(pathToAnnotatedFiles)
			rawFiles = openDirectory(pathToRaw)

			findFiles(coreNLPFiles, annotatedFiles, rawFiles, nameTxtOutput, verbCuesFile)

		else:
			print usageMessage

	else:
		print usageMessage


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


