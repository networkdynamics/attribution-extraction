import sys
import os
from intermediaries.nlpReaders.parc_reader import AnnotatedText as B
from parc_reader import ParcCorenlpReader as P
from nltk.tree import *
import csv
import multiprocessing
from multiprocessing import Manager
import time

#fixes all the omnipresent unicode issues
print sys.getdefaultencoding()
reload(sys)
sys.setdefaultencoding('utf-8')

#change this if you would only like to do a certain number of files, useful for testing
maxNumFiles = -1
minNumFile = 0

#base dir for all data files
data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/'))


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

'''
with open(data_dir + '/TrainKNNVerbPredictions.csv') as f:
	reader = csv.reader(f)
	knnPredictionsTrain = list(reader)
'''
knnPredictionsTrain = []


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

	for verb in enumerate(verbsList):
		metadata = verb[1][-1].split('=')[-1]
		metadata = metadata.split(';')
		sentID =  metadata[0]
		tokID = metadata[1]
		fileName = metadata[2]
		verbWord = verb[1][0].split('=')[-1]
		label = verb[1][-2].split('=')[-1]
		newVerbsList.append([verbWord, int(sentID), tokID, fileName, label])

	return newVerbsList

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in xrange(0, len(l), n))

def workerFunction(myFile, listOfAnnotatedFiles, listOfRawFiles, output, verbList, return_list):

	flagNoLabels = False

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

	#print('opening file: ' + myFile + ' ' + str(j) + ' out of ' + str(files))
	
	#extract the verbs whose metadata filename matches this NLP file
	specificFileVerbs = []
	for verb in verbList:
		if (verb[3] == filename):
			specificFileVerbs.append(verb)

	#open the file, extract the features and return all the rows
	fileRows = openFile(myFile, myAnnotatedFile, myRawFile, specificFileVerbs)
	return_list += fileRows




#get the files, open them, extract verbs and features and create a large array of rows
def findFiles1(listOfNLPFiles, listOfAnnotatedFiles, listOfRawFiles, output, verbCuesFile, flagTraining):

	verbList = openVerbCues(verbCuesFile)

	splitLists = list(chunks(listOfNLPFiles, len(listOfNLPFiles)/10))
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
		if i == 1:
			break
		for thisList in splitLists:
			myFile = thisList[i]
			p = multiprocessing.Process(target = workerFunction, args=(myFile, listOfAnnotatedFiles, listOfRawFiles, output, verbList, return_list))
			jobs.append(p)
			p.start()
		time.sleep(3)


	

	#append the files from last list (remainder of total files divided by 10)
	for myFile in lastList:
		p = multiprocessing.Process(target = workerFunction, args=(myFile, listOfAnnotatedFiles, listOfRawFiles, output, verbList, return_list))
		jobs.append(p)
		p.start()

	for proc in jobs:
		proc.join()

	open(os.path.join(data_dir, output), 'w').close()
	writeToTXT(return_list, os.path.join(data_dir, output), flagNoLabels)


def findFiles(listOfNLPFiles, listOfAnnotatedFiles, listOfRawFiles, output, verbCuesFile, flagTraining):

	verbList = openVerbCues(verbCuesFile)

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

		specificFileVerbs = []
		for verb in verbList:
			if (verb[3] == filename):
				specificFileVerbs.append(verb)

		
		fileRows, article = openFile(myFile, myAnnotatedFile, myRawFile, specificFileVerbs, flagTraining)
		myRows += fileRows
		print 'length'
		print len(fileRows)

		j = j + 1

		if j == maxNumFiles:
			break



	open(os.path.join(data_dir, output), 'w').close()
	writeToTXT(myRows, os.path.join(data_dir, output), flagNoLabels)

def openFile(coreNLPFileName, annotatedFileName, raw_file, verbList, flagTraining):

	allrows = []
	#open annotated if it exists
	if annotatedFileName != None:
		try:
			parc_xml = open(annotatedFileName).read()
			corenlp_xml = open(coreNLPFileName).read()
			raw_text = open(raw_file).read()
			article = P(corenlp_xml, parc_xml, raw_text)
			if flagTraining == True:
				filename = coreNLPFileName.split('/')[-1]
				rows = findFeatures(filename, article, verbList, annotatedFileName)
				return rows, article
			else:
				rows = findFeatures(filename, article, verbList, annotatedFileName)
				return rows, article


		except Exception, msg:
			print 'error opening file'
			print msg
			raise
			return allrows, None

	else:
		parc_xml = None
		corenlp_xml = open(coreNLPFileName).read()
		raw_text = open(raw_file).read()
		article = P(corenlp_xml, parc_xml, raw_text)

	filename = coreNLPFileName.split('/')[-1]

	rows = findFeatures(filename, article, realVerbList, annotatedFileName)
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
def findFeatures(filename, article, verbsList, annotatedFileName):

	rows = []

	openQuotes = False

	lastLabel = 'O'

	#'''
	specificFilePredictions = []
	for verb in knnPredictionsTrain:
		if (verb[3] == filename and verb[-1] == 'Y'):
			specificFilePredictions.append(verb)

	#'''
	sentencesToSkip = []
	if annotatedFileName != None:
		attributions = article.attributions
		for attr in attributions:
			content = attributions[attr]['content']
			source = attributions[attr]['source']

			if len(content) < 2:
				for token in content:
					sentencesToSkip.append(token['sentence_id'])
				article.remove_attribution(attr)

				continue
			if len(source) == 0:
				for token in content:
					sentencesToSkip.append(token['sentence_id'])
				article.remove_attribution(attr)
				continue
			if len(source) > 0:
				firstChar = content[0]['character_offset_begin']
				lastChar = content[-1]['character_offset_end']
				sourceFirstChar = source[0]['character_offset_begin']

				if sourceFirstChar > firstChar and sourceFirstChar < lastChar:
					for token in content:
						sentencesToSkip.append(token['sentence_id'])
					continue

			if content[0]['pos'] in [',', '.', '``', '"', '?', '!']:
				content = content[1:]
			if content[-1]['pos'] in [',', '.', '``', '"', '?', '!']:
				content = content[:-1]

			attributions[attr]['content'] = content

	sentencesToSkip = list(set(sentencesToSkip))

	#begin extracting features
	for sentence in article.sentences:

		if len(sentence['tokens']) > 70:
			continue

		if sentence['id'] in sentencesToSkip:
			continue
		
		#'''
		skip = False
		for verb in specificFilePredictions:
			if int(verb[1]) == sentence['id']:
				skip = True
				for token in sentence['tokens']:
					if token['attribution'] != None:
						skip = False
						break
				break

		if skip == True:
			continue
		#'''
		
		lengthOfSentence = len(sentence['tokens'])
		idOfSentence = sentence['id']

		currentSentPers = 'false'
		

		#these are the features that will stay the same across each token in the sentence
		#e.g. containsOrganization, containsNamedEntity etc.
		#returns the array for this sentence of tokens, pos, lemma which will be used to
		#constitute the token based features
		rowSentFeats = ''
		tokenArray, posArray, lemmaArray,  peopleHyponym, orgHyponym, rowSentFeats = \
						getSentenceFeatures(sentence, rowSentFeats)

		#get verb cue, sentence wide features, e.g. containsVerbCue, verbCueNearTheEnd?
		rowSentFeats, verbCueIds = getVerbListSentenceFeatures(sentence, verbsList, rowSentFeats)


		#rowSentFeats now contains a string that looks like
		#'containsOrg='True'\tcontainsVerbCue='True'\t.....'
		prevSyntactic = ''
		
		


		for token in sentence['tokens']:
			row = rowSentFeats

			word = str(token['word']).lower()
			pos = str(token['pos'])
			lemma = str(token['lemma'])
			idOfToken = token['id']		

			if annotatedFileName != None:
				#assign labels
				label = token['attribution']

				role = token['role']

				if label == None or role != 'content':
					row = 'O\t' + row
					lastLabel = 'O'
				elif role == 'content' and lastLabel == 'O':
					row = 'B\t' + row
					lastLabel = 'B'
				elif role == 'content':
					row = 'I\t' + row
					lastLabel = 'I'
				else:
					row = 'O\t' + row
					lastLabel = 'O'
			else:
				row = '\t' + row
			
			thisChanged = False
			if "''" in str(token['word']) and openQuotes == True:
				openQuotes = False
				thisChanged = True

			#append the features
			row += 'sentenceLength=' + str(lengthOfSentence) + '\t'


	
			row = getTokenFeatures(idOfToken, tokenArray, row, 'word')
			row = getTokenFeatures(idOfToken, posArray, row, 'pos')
			row = getTokenFeatures(idOfToken, lemmaArray, row, 'lemma')

			row = getTokenPairs(idOfToken, tokenArray, row, 'word')
			row = getTokenPairs(idOfToken, posArray, row, 'pos')
			row = getTokenPairs(idOfToken, lemmaArray, row, 'lemma')

			row = findRelations(token, row)

			peopleHyponym = list(set(peopleHyponym))
			for (idHypo, hyponym) in peopleHyponym:
				row += 'personHyponym=' + str(idHypo - idOfToken)+ '\t'

			orgHyponym = list(set(orgHyponym))	
			for (idHypo, hyponym) in orgHyponym:
				row += 'organizationHyponym=' + str(idHypo - idOfToken) + '\t'

			#the inside quote labels
			if openQuotes == True:
				row += "insidequotes='true'\t"


			if ("``" in str(token['word'])) or ("''" in str(token['word']) and openQuotes == False and thisChanged == False):
				openQuotes = True

			prevSyntactic, row = getConstituentLabels(token, sentence, row, verbsList, filename, prevSyntactic)
			getVerbDependencies(token, sentence, row, verbsList, filename)

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
	idVerbCueTokens = []

	foundAccording = False
	for token in sentence['tokens']:

		word = token['word']

		if (word.lower() != 'to' and foundAccording == True):
			foundAccording = False
		if (word.lower() == 'according'):
			foundAccording = True

		if (word.lower() == 'to' and foundAccording == True):

			rowSentFeats += '''containsVerbCue='true'\t'''
			idVerbCueTokens.append(token['id'])
			foundAccording = False
			if token['id'] >= nearEnd:
				rowSentFeats += '''verbCueNearEnd='true'\t'''

	for verb in verbs:
		if str(verb[1]) == str(sentenceID):
			sentVerb = sentence['tokens'][int(verb[2])]

			if sentVerb['word'] == verb[0]:
				if verb[4] == 'Y':
					containsVerbCue = True
					rowSentFeats += '''containsVerbCue='true'\t'''
					idVerbCueTokens.append(verb[2])
					if int(verb[2]) >= nearEnd:
						rowSentFeats += '''verbCueNearEnd='true'\t'''


					parent = sentVerb['parents']
					if parent == []:
						continue
					parentToken = parent[0][1]
					for newVerb in verbs:
						if parentToken['sentence_id'] == int(newVerb[1]) and parentToken['id'] == int(newVerb[2]) and newVerb[-1] == 'Y':
							rowSentFeats += '''containsNestedVerbCue='true'\t'''
	return rowSentFeats, idVerbCueTokens


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
	foundNoun = False

	foundAny = False

	for token in sentence['tokens']:

			if str(token['ner']) == 'PERSON':
				word = 'PERSON'
			elif str(token['ner']) == 'ORGANIZATION':
				word = 'ORGANIZATION'
			elif str(token['ner']) == 'LOCATION':
				word = 'LOCATION'
			else:
				word = str(token['word'])

			namedEnt = str(token['ner'])

			if (namedEnt == 'PERSON' and foundPerson==False):
				row += '''containsPerson='true'\t'''
				foundPerson = True
			elif (namedEnt == 'ORGANIZATION' and foundOrganization==False):
				row += '''containsOrganization='true'\t'''
				foundOrganization = True
			pos = str(token['pos'])
			if 'PRP' == pos and foundPronoun==False:
				row += '''containsPronoun='true'\t'''
				foundPronoun=True

			if "''" in word and foundQuotes==False:
				row += '''containsQuotes='true'\t'''
				foundQuotes=True

			


			tokenArray = tokenArray + [word.lower()]
			posArray = posArray + [pos]
			lemmaArray = lemmaArray + [str(token['lemma'])]

			#get positions of possible hyponyms within sentence
			for hypo in peopleHyponyms:
				if word.lower() == hypo:
					peopleHyponym.append((token['id'], word))

			for hypo in orgHyponyms:
				if word.lower() == hypo:
					orgHyponym.append((token['id'], word))
			
			if (word.lower() != 'to' and foundAccording == True):
				foundAccording = False
			if (word.lower() == 'according'):
				foundAccording = True
			if (word.lower() == 'to' and foundAccording == True):
				row += '''containsAccordingTo='true'\t'''
				foundAccording = False

			if token['lemma'] in nounCues and token['pos'].startswith('N') and foundNoun == False:
				row += '''containsNounCue='true'\t'''
				foundNoun = True

			if (foundPerson == True or foundOrganization == True or foundPronoun == True or foundQuotes == True or foundAccording == True or foundNoun == True) and foundAny == False:
				row += '''foundAnySentenceFeatures='true'\t'''
				foundAny = True



	return tokenArray, posArray, lemmaArray, peopleHyponym, orgHyponym, row


def getConstituentLabels(token, sentence, row, verbsList, filename, prevSyntactic):
	rootSentence = sentence['c_root']

	listLabels = []
	listConstituencies = []
	highestConstituentWithLeftMost = 100

	#get all parent constituents that include it
	s = Stack()
	if 'c_parent' not in token:
		print 'NO C PARENT'
		return prevSyntactic, row

	s.push(token['c_parent'])

	while not s.isEmpty():
		currentParent = s.pop()
		if 'c_tag' in currentParent:
			listLabels.append((currentParent['c_tag'], currentParent['c_depth']))
			listConstituencies.append((get_constituent_tokens(currentParent), currentParent['c_depth'], currentParent['c_tag'], currentParent))
		if 'c_parent' in currentParent and currentParent['c_parent'] != None:
			s.push(currentParent['c_parent'])

	for (tag, depth) in listLabels:
		row += 'const=(' + tag + ',' + str(depth) + ')\t'	

	currentAddition = (token['c_tag'], token['c_depth'], [token], token)
	maxDepth = token['c_depth']
	for elem in listConstituencies:
		tokens = elem[0]
		depth = elem[1]
		tag = elem[2]
		constituent = elem[3]
		if tokens[0]['sentence_id'] == token['sentence_id'] and tokens[0]['id'] == token['id'] and depth < maxDepth:
			currentAddition = (tag, depth, tokens, constituent)

	verbCueInConstituents = False
	for elem in currentAddition[2]:
		verbVersion = [elem['word'], str(elem['sentence_id']), str(elem['id']), filename, 'Y']
		if verbVersion in verbsList:
			verbCueInConstituents = True
		if verbVersion[0].lower() == 'according':
			verbCueInConstituents = True

	row += 'constLabel=' + currentAddition[0] + '\t' + 'constSpanLength=' + str(len(currentAddition[2])) + '\t' + 'depthSpan=' + str(currentAddition[1]) + '\t'

	if currentAddition[1] != 0:
		parentConstituent = currentAddition[3]['c_parent']
		parentTokens = get_constituent_tokens(parentConstituent)
		parentAddition = (parentConstituent['c_tag'], parentConstituent['c_depth'], len(parentTokens))
		row += 'parentConstLabel=' + parentConstituent['c_tag'] + '\t' + 'parentConstSpanLength=' + str(len(parentTokens)) + '\t' + 'parentDepthSpan=' + str(parentConstituent['c_depth']) + '\t'
	
		for elem in parentTokens:
			verbVersion = [elem['word'], str(elem['sentence_id']), str(elem['id']), filename, 'Y']
			if verbVersion in verbsList:
				verbCueInConstituents = True
			if verbVersion[0].lower() == 'according':
				verbCueInConstituents = True


		if verbCueInConstituents == True:
			row += "parentConstSpanVC='true'\t"

	return prevSyntactic, row


def get_constituent_tokens(constituent, recursive=True):

	tokens = []
	for child in constituent['c_children']:
		if 'pos' in child:
		    tokens.append(child)
		elif recursive:
		    tokens.extend(get_constituent_tokens(child, recursive))

	return tokens



def getVerbDependencies(token, sentence, row, verbsList, filename):

	parent = token['parents']
	found = False
	depth = 0
	while parent != []:
		parentTok = parent[0][1]
		if parentTok['word'].lower() == 'according':
			row += "dependentOnAccording='true'\t"
			break
		for verb in verbsList:
			if int(verb[1]) == parentTok['sentence_id'] and int(verb[2]) == parentTok['id'] and verb[-1] == 'Y':
				row += "dependentOnVerbCue='true'\t"
				found = True
				for newVerb in verbsList:
					if token['sentence_id'] == int(newVerb[1]) and int(newVerb[2]) == token['id'] and newVerb[-1] == 'Y' and depth == 0:
						row += "verbCueDependentOnVerbCue='true'\t"
				break
		if found == True:
			break
		else:
			parent = parentTok['parents']
			depth += 1









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
			pathToCORENLP = '/home/ndg/dataset/ptb2-corenlp/CoreNLP/train/'
			pathToAnnotatedFiles = '/home/ndg/dataset/parc3/train/'
			pathToRaw = '/home/ndg/dataset/ptb2-corenlp/masked_raw/train/'

			verbCuesFile = 'PARCTrainVerbFeats.csv'
			nameTxtOutput = 'PARCTrainContentSpans3.txt'
			coreNLPFiles = openDirectory(pathToCORENLP)
			annotatedFiles = openDirectory(pathToAnnotatedFiles)
			rawFiles = openDirectory(pathToRaw)

			findFiles(coreNLPFiles, annotatedFiles, rawFiles, nameTxtOutput, verbCuesFile, True)

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


