import sys
import os
from SETTINGS import PACKAGE_DIR
from parc_reader import ParcCorenlpReader as P
import verbCuesFeatureExtractor as verbCuesIdentity
from nltk.tree import *
import csv
from operator import itemgetter
from collections import defaultdict
from itertools import groupby
import pdb

REPORTING_VERBS_CSV_PATH = os.path.join(
    PACKAGE_DIR, 'source', 'intermediaries', 'reportingVerbs.csv')

#fixes all the omnipresent unicode issues
print sys.getdefaultencoding()
reload(sys)
sys.setdefaultencoding('utf-8')

#change this if you would only like to do a certain number of files, useful for testing
maxNumFiles = -1

miscList = []
acceptableNERS = ['PERSON', 'ORGANIZATION', 'MISC', 'LOCATION', 'PERCENT']
unacceptablePOSs = ['MD', 'RB', 'JJ', 'JJR']



#base dir for all data files
data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/'))

with open(data_dir + '/allHypos.csv', 'rb') as f:
    reader = csv.reader(f)
    hypos = list(reader)


#with open(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'allHypos.csv')), 'rb') as f:
#    reader = csv.reader(f)
#    hypos = list(reader)

nominals = hypos[0]
nominals = nominals + ['many', 'both', 'some', 'all', 'one']
nominals = sorted(nominals)

print len(nominals)

tupleNominals = list(set([(words.split(' ')[0], words) for words in nominals]))
firstNominals = [nominal for (nominal, phrase) in tupleNominals]

with open(REPORTING_VERBS_CSV_PATH, 'r') as f:
	listVerbs = []
	string = f.readlines()[0]
	listVerbs = string.split(' ')

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

def findFiles(coreNlPFiles, annotatedFiles, rawFiles, predictedQuotes):

	#open each NLP File
	j = 0
	myRows = []
	for myFile in coreNlPFiles:

		#in case you want a minimum file number
		if j < -1:
			j = j + 1
			continue
		else:
			files = len(coreNlPFiles)
			filename = myFile.split('/')[-1]
			fileNoXML = filename.split('.xml')[0]

			print fileNoXML
			print filename

			myAnnotatedFile = []

			#extract the PARC filename that match the title of the NLP filename 
			if annotatedFiles != None:
				myAnnotatedFile = [s for s in annotatedFiles if filename in s]
				if len(myAnnotatedFile) == 1:
					myAnnotatedFile = myAnnotatedFile[0]
				else:
					#didn't find a file
					print 'error opening Annotated File'
					continue


			myRawFile = [s for s in rawFiles if fileNoXML in s][0]

			
			print('opening file: ' + myFile + ' ' + str(j) + ' out of ' + str(files))
			

			#open the file, extract the features and return all the rows
			if myAnnotatedFile != []:
				fileRows, article = openFile(myFile, myAnnotatedFile, myRawFile, None)
			else:
				fileRows, article = openFile(myFile, None, myRawFile, predictedQuotes)

			myRows += fileRows

			numTokens = len(myRows)

			j = j + 1

			if j == maxNumFiles:
				break

	return myRows

def openFile(coreNLPFileName, annotatedFileName, rawFile, predictedQuotes):

	rows = []
	filename = coreNLPFileName.split('/')[-1]


	#open annotated if it exists
	if annotatedFileName != None:

		try:
			corenlp_xml = open(coreNLPFileName).read()
			parc_xml = open(annotatedFileName).read()
			raw_text = open(rawFile).read()

			article = P(corenlp_xml, parc_xml, raw_text)
			try:

				filerows, newArticle = process(article, filename, None, None)
				rows += filerows
			except Exception, e:
				raise 
		except:
			print 'error opening file'
			raise
			return rows, []

	else:
		try:
			parc_xml = None
			corenlp_xml = open(coreNLPFileName).read()
			raw_text = open(rawFile).read()

			article = P(corenlp_xml, parc_xml, raw_text)
			filerows, newArticle = process(article, filename, predictedQuotes, [])
			rows += filerows
		except:
			print 'error opening file'
			raise
			return rows, []

	return rows, newArticle	


def process(article, filename, predictedQuotes, verbCues):
	print '%d paragraphs in this article' % len(article.paragraphs)

	processedArticle = []


	if predictedQuotes != None and predictedQuotes != []:
		#article = addAttrs(predictedQuotes, article, filename)
		article = addAttrsList(predictedQuotes, article, filename)

	if verbCues == None:
		verbCues = []

		for attr in article.attributions:
			currentAttr = article.attributions[attr]
			if currentAttr['cue'] != []:
				for token in currentAttr['cue']:
					if token['pos'].startswith('V'):
						verbCues.append([token['word'], str(token['sentence_id']), str(token['id']), 'BLAH', 'Y'])
					else:
						continue



	attributionId = 0

	for sentence in article.sentences:
		paragraphID = sentence['paragraph_idx']
		sentenceID = sentence['id']

		thisSent = processedSentence()
		thisSent['sentence_id'] = sentence['id']
		thisSent['article_id'] = filename
		thisSent['mentions'] = sentence['mentions']
		thisSent['parag_id'] = paragraphID

		lengthSentence = len(sentence['tokens'])

		currEntityStart = 0
		currEntityStop = 0

		tokenInQuotes = []

		for token in sentence['tokens']:
			if 'attribution' not in token:
				token['attribution'] = None
				token['role'] = None

			attr = token['attribution']
			tokID = token['id']
			word = token['word']
			lemma = token['lemma']
			role = token['role']
			pos = token['pos']
			ner = token['ner']
			prevToken = sentence['tokens'][token['id'] - 1]
			token['parag_id'] = paragraphID
			

			if attr is not None and role == 'content' and len(attr['content']) < 100:
				if (token['id'], token['sentence_id']) not in tokenInQuotes:
					attrId = attr['id']
					quote = createQuotes(attr)
					quote['attribution_id'] = attrId
					quote['sentence_id'] = sentenceID
					quote['parag_id'] = paragraphID

					thisSent.addToken(quote)

					for token in quote['tokenArray']:
						tokenInQuotes.append((token['id'], token['sentence_id']))

			elif ('NNP' in pos and (ner in acceptableNERS or ner == None)) or (ner in acceptableNERS):
					if tokID > currEntityStart and tokID <= currEntityStop:
						continue
					else:
						entity, currentToken = createEntity(token, sentence)
						currEntityStart = entity['token:start']
						currEntityStop = entity['token:stop']
						thisSent.addToken(entity)
						#SourceSpeakerEntity : ['tokenArray', 'attribution_id', 'token:start', 'token:stop', 'sentence_id', 'parag_id']

			elif (pos == 'PRP' or (pos.startswith('W') and pos != 'WRB' and lemma != 'what')) or lemma == 'those':
				entity = SourceSpeakerEntity()
				entity['tokenArray'] = [token]
				if attr != None:
					entity['attribution_id'] = attr['id']
				entity['token:start'] = tokID
				entity['token:stop'] = tokID
				entity['sentence_id'] = sentenceID
				entity['parag_id'] = paragraphID
				thisSent.addToken(entity)

			elif pos.startswith('V'):

				found = False
				if word.lower() == 'according':
					if tokID + 1 < lengthSentence:
						if sentence['tokens'][tokID + 1]['lemma'] == 'to':
							parag_id = sentence['paragraph_idx']
							verb = createAccordingTo([token, sentence['tokens'][tokID + 1]], parag_id, lengthSentence)
							thisSent.addToken(verb)
							found = True
							continue

				for verb in verbCues:
					if verb[1] == str(sentenceID) and verb[2] == str(tokID) and verb[4] == 'Y':
						parag_id = sentence['paragraph_idx']
						verb = createVerb(token, parag_id)
						thisSent.addToken(verb)
						found = True
						break

				if found == False:
					thisSent.addToken(token)
					continue

			elif lemma in firstNominals and (pos.startswith('N') or pos == 'JJR' or pos == 'JJS'):
				mynominals = [(first, phrase) for (first, phrase) in tupleNominals if first == lemma]
				nominal = nominalEntites(token, sentence, mynominals)
				if nominal == None:
					thisSent.addToken(token)
				else:
					thisSent.removeLastToken(nominal)
					thisSent.addToken(nominal)

			elif pos.startswith('R') or pos.startswith('J'):
				continue
			else:
				thisSent.addToken(token)

		foundSignObject = False
		if thisSent['words'] == None:
			continue
		for token in thisSent['words']:
			if 'pos' in token:
				continue
			elif token.truncated() == '<ENTITY>' or token.truncated() == '<QUOTE>':
				foundSignObject = True

		if foundSignObject == False:
			continue
		else:
			processedArticle.append(thisSent)


		


	return processedArticle, article


#['tokenArray', 'attribution_id', 'token:start', 'token:stop', 'sentence_id', 'parag_id']
#verb: ['tokenArray', 'attribution_id', 'token:start', 'token:stop', 'sentence_id', 'parag_id']


#['attribution', 'word', 'character_offset_begin', 'character_offset_end', 'pos', 'lemma', 'sentence_id', 'entity_idx', 'speaker', 'parents', 'role', 'ner', 'id']
#['paragraph_idx', 'tokens', 'entities', 'attributions', 'references', 'mentions', 'root', 'id']

def checkNextLabel(sentID, tokID, article, fileQuotes, row):
	sentence = article.sentences[sentID]['tokens']
	currentToken = article.sentences[sentID]['tokens'][tokID]

	lengthSentence = len(sentence)

	leeway = lengthSentence - tokID

	try:
		pos = fileQuotes.index(row)
	except:
		return False

	for i in range(3):
		if tokID < lengthSentence - i:
			nextToken = fileQuotes[pos + i]
			lastToken = fileQuotes[pos - 1]

			split = nextToken.split('\t')
			splitLast = lastToken.split('\t')

			label = row[0]
			metadata = split[1].split(';')
			metadataLast = splitLast[1].split(';')

			nextSentID = int(metadata[1].split('=')[1])
			nextTokenID = int(metadata[2].split('=')[1])
			lastSentId = int(metadataLast[1].split('=')[1])

			if lastSentId != sentID:
				return False
			if nextToken.startswith('I') or nextToken.startswith('B'):
				return True
			if article.sentences[nextSentID]['tokens'][nextTokenID]['pos'].startswith('.'):
				return False

	return False

def addAttrsList1(predictedQuotes, article, articleFilename):
	span = []
	spans = []
	lastLabel = 'O'
	lastFilename = ''
	attrID = 0

	print articleFilename

	for row in predictedQuotes:
		split = row.split('\t')

		label = row[0]
		metadata = split[1].split(';')

		sentID = int(metadata[1].split('=')[1])
		tokID = int(metadata[2].split('=')[1])

		if (row == predictedQuotes[-1] and len(span) > 1):
			attrSpan = [article.sentences[span[0][1]]['tokens'][span[0][0]]]
			del span[0]
			length = 1
			for token in span:
				thisTOKID = token[0]
				thisSENTID = token[1]
				attrSpan.append(article.sentences[thisSENTID]['tokens'][thisTOKID])
			else:
				article.add_attribution(
					cue_tokens=[],
					content_tokens= attrSpan,
					source_tokens = [],
					id_formatter = 'my_attribution_' + str(attrID) + '_' + articleFilename,
					)
			attrID = attrID + 1
			span = []

		elif (label == 'B' and lastLabel == 'O'):# or (label == 'I' and lastLabel == 'O'):
			if len(span) != 0:
				if len(span)<=1:
					span = []
					lastLabel = 'O'
					continue


				attrSpan = [article.sentences[span[0][1]]['tokens'][span[0][0]]]
				del span[0]
				length = 1
				for token in span:
					thisTOKID = token[0]
					thisSENTID = token[1]
					attrSpan.append(article.sentences[thisSENTID]['tokens'][thisTOKID])
				else:
					article.add_attribution(
						cue_tokens=[],
						content_tokens= attrSpan,
						source_tokens = [],
						id_formatter = 'my_attribution_' + str(attrID) + '_' + articleFilename,
						)
				attrID = attrID + 1
				span = []

			span = [(tokID, sentID, label)]
			lastLabel = 'B'
			continue
		elif label == 'O' and (lastLabel == 'I' or lastLabel == 'B') and checkNextLabel(sentID, tokID, article, predictedQuotes, row):
			span.append((tokID, sentID, label))
			lastLabel = 'I'
		elif label == 'I' and lastLabel == 'O':
			lastLabel = 'O'
			continue
		elif label == 'I' or label == 'B':
			span.append((tokID, sentID, label))
			lastLabel = 'I'
		elif label == 'O':
			lastLabel = 'O'
			continue
		else:
			continue

	return article


def addAttrsList(predictedQuotes, article, articleFilename):


	span = []
	spans = []
	lastLabel = 'O'
	lastFilename = ''
	attrID = 0

	print articleFilename

	for row in predictedQuotes:
		split = row.split('\t')

		label = row[0]
		metadata = split[1].split(';')

		sentID = int(metadata[1].split('=')[1])
		tokID = int(metadata[2].split('=')[1])

		if label == 'I' and lastLabel == 'O':
			print 'DIDN"T WORK DIDNT WORK'

		if (row == predictedQuotes[-1] and len(span) > 1):
			attrSpan = [article.sentences[span[0][1]]['tokens'][span[0][0]]]
			del span[0]
			length = 1
			for token in span:
				thisTOKID = token[0]
				thisSENTID = token[1]
				attrSpan.append(article.sentences[thisSENTID]['tokens'][thisTOKID])
			else:
				article.add_attribution(
					cue_tokens=[],
					content_tokens= attrSpan,
					source_tokens = [],
					id_formatter = 'my_attribution_' + str(attrID) + '_' + articleFilename,
					)
			attrID = attrID + 1
			span = []


		elif (label == 'B'):
			if len(span) != 0:
				if len(span)<=1:
					span = []
					lastLabel = 'O'
					continue


				attrSpan = [article.sentences[span[0][1]]['tokens'][span[0][0]]]
				del span[0]
				length = 1
				for token in span:
					thisTOKID = token[0]
					thisSENTID = token[1]
					attrSpan.append(article.sentences[thisSENTID]['tokens'][thisTOKID])
				else:
					article.add_attribution(
						cue_tokens=[],
						content_tokens= attrSpan,
						source_tokens = [],
						id_formatter = 'my_attribution_' + str(attrID) + '_' + articleFilename,
						)
				attrID = attrID + 1
				span = []

			span = [(tokID, sentID, label)]
			lastLabel = 'B'
			continue
		elif label == 'I':
			span.append((tokID, sentID, label))
			lastLabel = 'I'
		elif label == 'O':
			lastLabel = 'O'
			continue
		else:
			continue

		


	return article

def createQuotes(attribution):
	content = attribution['content']

	ids = []
	for token in content:
		ids.append(token['id'])

	
	newQuote = Quote()
	newQuote['token:start'] = ids[0]
	newQuote['token:stop'] = ids[-1]
	newQuote['tokenArray'] = content

	return newQuote

#verb: ['tokenArray', 'attribution_id', 'token:start', 'token:stop', 'sentence_id', 'parag_id']

def createVerb(token, parag_id):
	newVerb = reportingVerb()
	attr = token['attribution']

	newVerb['tokenArray'] = [token]
	if attr != None:
		newVerb['attribution_id'] = attr['id']
	else:
		newVerb['attribution_id'] = None

	newVerb['token:start'] = token['id']
	newVerb['token:stop'] = token['id']
	newVerb['sentence_id'] = token['sentence_id']
	newVerb['parag_id'] = parag_id

	return newVerb

def createAccordingTo(tokens, parag_id, lengthSentence):
	newVerb = reportingVerb()
	attr = tokens[0]['attribution']

	newVerb['tokenArray'] = tokens
	if attr != None:
		newVerb['attribution_id'] = attr['id']
	else:
		newVerb['attribution_id'] = None

	newVerb['token:start'] = tokens[0]['id']
	newVerb['token:stop'] = tokens[1]['id']
	newVerb['sentence_id'] = tokens[0]['sentence_id']
	newVerb['parag_id'] = parag_id

	return newVerb

#SourceSpeakerEntity : ['tokenArray', 'attribution_id', 'token:start', 'token:stop', 'sentence_id', 'parag_id']

def createEntity(token, sentence):

	thisEntity = SourceSpeakerEntity()

	currentTokenID = token['id']
	thisEntity['token:start'] = currentTokenID
	thisEntity['entity_idx'] = token['entity_idx']

	if token['attribution'] != None:
		thisEntity['attribution_id'] = token['attribution']['id']
	thisEntity['sentence_id'] = sentence['id']
	thisEntity['parag_id'] = sentence['paragraph_idx']

	lengthSentence = len(sentence['tokens'])



	if currentTokenID > 0:
		prevToken = sentence['tokens'][currentTokenID - 1]
		if prevToken['pos'] == 'DT':
			thisEntity.addTokenToList(prevToken)


	thisEntity.addTokenToList(token)
	currentTokenID += 1
	if currentTokenID < lengthSentence:
		nextToken = sentence['tokens'][currentTokenID]
	else:
		thisEntity['token:stop'] = currentTokenID - 1
		return thisEntity, currentTokenID - 1


	while ('NNP' in nextToken['pos'] and (nextToken['ner'] in acceptableNERS or nextToken['ner'] == None)) or  (nextToken['ner'] in acceptableNERS):
		thisEntity.addTokenToList(nextToken)
		currentTokenID += 1
		if currentTokenID < lengthSentence:
			nextToken = sentence['tokens'][currentTokenID]
			if ('NNP' in nextToken['pos'] and (nextToken['ner'] in acceptableNERS or nextToken['ner'] == None)) or  (nextToken['ner'] in acceptableNERS):
				continue	
			elif currentTokenID	< lengthSentence - 1:
				nextNextToken = sentence['tokens'][currentTokenID+1]


			else:
				currentTokenID = currentTokenID - 1
		else:
			break

	thisEntity['token:stop'] = currentTokenID

	
	return thisEntity, currentTokenID


#SourceSpeakerEntity : ['tokenArray', 'attribution_id', 'token:start', 'token:stop', 'sentence_id', 'parag_id']

def nominalEntites(token, sentence, mynominals):

	match, currID, pos = findNominals(token, sentence, mynominals)

	#if sentence['id'] == 52:
	#	pdb.set_trace()

	if match == False:
		return None

	prevTokenID = currID - 1

	acceptableDETs = ['DT', 'PRP$']
	acceptableMODs = ['JJS', 'JJ', 'JJR', 'POS', 'RB', 'RBR', 'RBS', 'NN', 'IN', 'CD']

	additions = False

	while prevTokenID >= 0:
		partOfSpeech = sentence['tokens'][prevTokenID]['pos']

		if partOfSpeech in acceptableMODs and prevTokenID == 0:
			additions = True
			break

		elif partOfSpeech in acceptableMODs :
			prevTokenID = prevTokenID - 1
			additions = True

		elif partOfSpeech in acceptableDETs:
			additions = True
			break
		else:
			prevTokenID = prevTokenID + 1
			break


	newEntity = SourceSpeakerEntity()

	tokenRange = currID
	newEntity['token:start'] = currID
	newEntity['token:stop'] = pos
	newEntity['sentence_id'] = sentence['id']
	newEntity['parag_id'] = sentence['paragraph_idx']


	if additions == True:
		tokenRange = prevTokenID
		newEntity['token:start'] = prevTokenID

	while tokenRange != pos + 1:
		newEntity.addTokenToList(sentence['tokens'][tokenRange])
		tokenRange += 1

	if token['attribution'] != None:
		newEntity['attribution_id'] = token['attribution']['id']

	return newEntity


def findNominals(token, sentence, mynominals):

	currID = token['id']

	match = False

	listNominals = [phrase.split() for (first, phrase) in mynominals]
	listNominals.sort(key = len, reverse = True)

	for (first, phrase) in mynominals:
		pos = currID

		listPhrase = phrase.split()
		lengthPhrase = len(listPhrase)
		if lengthPhrase > 1:
			lengthPhrase = lengthPhrase - 1
			pos = currID + 1
			while lengthPhrase != 0 and pos < len(sentence['tokens']):
				if sentence['tokens'][pos]['lemma'] != listPhrase[pos - currID]:
					break
				elif lengthPhrase == 1:
					match = True
					return match, currID, pos
				else:
					pos = pos + 1
					lengthPhrase = lengthPhrase - 1
		else: #phrase is one token
			match = True
			return match, currID, pos

	return False, 0 , 0				

def main():
	usageMessage = '\nCorrect usage of the Entity Source Feature Extractor command is as follows: \n' + \
					'\n\n WHEN AN ANNOTATED FILESET EXISTS TO GET LABELS FROM:\n' + \
					'To extract tokens and their features: \n python source/sourceEntityTrainingSet.py -labelled /pathToCoreNLPDirectory /pathToAnnotatedFilesDirectory /pathToRawDirectory \n' + \
					'\n\n WHEN THE LABELS ARE UNKNOWN:\n' + \
					'To extract tokens and their features: \n python source/sourceEntityTrainingSet.py -unlabelled /pathToCoreNLPDirectory /pathToRawDirectory /pathToPredictedQuotes \n' + \
					'\nFor reference, the path to the CoreNLP file is: /home/ndg/dataset/ptb2-corenlp/CoreNLP/ + train, test or dev depending on your needs. \n' + \
					'The path to the Parc3 files is /home/ndg/dataset/parc3/ + train, test or dev depending on your needs.\n'

	args = sys.argv

	if len(args) == 5:

		flag = args[1]
		

		if flag == '-labelled':
			pathToCORENLP = args[2]
			pathToAnnotatedFiles = args[3] 
			pathToRaw = args[4]

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


			coreNLPFiles = openDirectory(pathToCORENLP)
			annotatedFiles = openDirectory(pathToAnnotatedFiles)
			rawFiles = openDirectory(pathToRaw)


			findFiles(coreNLPFiles, annotatedFiles, rawFiles, None)

		elif flag == '-unlabelled':
			flag = args[1]
			pathToCORENLP = args[2]
			pathToRaw = args[3]
			pathToPredictedQuotes = args[4]

			if flag != '-unlabelled':
				print usageMessage
				return

			if os.path.isdir(pathToCORENLP):
				print 'valid path to a directory'
			else:
				print 'ERROR: The path to this coreNLP directory does not exist.'
				print usageMessage
				return


			if os.path.isdir(pathToRaw):
				print 'valid path to a directory'
			else:
				print 'ERROR: The path to this raw directory does not exist.'
				print usageMessage
				return


			#with open(os.path.join(data_dir, pathToPredictedQuotes), 'r') as f:
			#	listQuotes = []
			#	listQuotes = f.readlines()


			quoteDict = defaultdict(list)
			lastFN = ''

			for row in listQuotes:
				split = row.split('\t')
				label = split[0]
				meta = split[1]

				metaSplit = meta.split(';')
				filename = metaSplit[0].split('=')[1]
				sentID = metaSplit[1].split('=')[1]
				tokID = metaSplit[2].split('=')[1]

				quoteDict[filename].append([sentID, tokID, label])
				

			coreNLPFiles = openDirectory(pathToCORENLP)
			rawFiles = openDirectory(pathToRaw)

			findFiles(coreNLPFiles, None, rawFiles, quoteDict)

		else:
			print usageMessage
			return




class processedSentence(dict):

	def __init__(self, *args, **kwargs):
		super(processedSentence, self).__init__(*args, **kwargs)
		mandatory_listy_attributes = [
			'words', 'article_id', 'sentence_id', 'entities', 'parag_id']
		for attr in mandatory_listy_attributes:
			if attr not in self:
				self[attr] = None


	def __str__(self):

		stringRep = ''
		if self['words'] == None:
			return ''

		for t in self['words']:
			if t.has_key('pos'):
				stringRep += t['word'] + ' '
			else:
				stringRep += t.truncated()

		stringRep = stringRep.encode('utf-8')
		return stringRep



	def __repr__(self):
		stringRep = ''
		if self['words'] == None:
			return ''

		for t in self['words']:
			if t.has_key('pos'):
				stringRep += t['word'] + ' '
			else:
				stringRep += t.truncated()

		stringRep = stringRep.encode('utf-8')
		return stringRep

	def addToken(self, token):

		
		if self['words'] == None:
			if 'pos' in token:
				self['words'] = [token]
			elif 'tokenArray' in token:
				self['words'] = [token]
		else:
			if 'pos' in token:
				self['words'].append(token)
			elif 'tokenArray' in token:
				self['words'].append(token)
			else:
				raise MyException("you tried adding a token to the list that isn't valid")

	def removeLastToken(self, nominal):
		words = self['words']

		nominalTokens = nominal['tokenArray']
		lengthTokens = len(nominalTokens)

		if words == None:
			return

		if len(words) == 1:
			if 'pos' in words[0] and nominalTokens[0] == words[0]:
				del words[0]
			elif 'pos' in words[0]:
				return
			elif words[0].truncated() == '<ENTITY>':
				prevEntityArray = words[0]['tokenArray']
				lenthPrevEntity = len(prevEntityArray)
				i = 0
				same = False

				while i < lenthPrevEntity and i < len(nominalTokens):
					if prevEntityArray[i] == nominalTokens[i]:
						same = True
						i += 1
					else:
						same = False
						break


				if same == True:
					del words[0]
			elif 'TokenArray' in words[0]:
				return
		elif len(words) > 1:

			prevObject = words[-1]
			i = 0
			while (i < lengthTokens):
				if 'pos' in prevObject and words[-1] in nominalTokens:
					del words[-1]
					#if len(words) == 1:
					#	del words[0]
				elif 'pos' in prevObject: 
					return
				elif prevObject.truncated() == '<ENTITY>':
					prevTokenArray = words[-1]['tokenArray']
					prevTokens = [word['word'] for word in prevTokenArray]
					theseTokens = [word['word'] for word in nominalTokens]

					if set(prevTokens) <= set(theseTokens):
						del words[-1]

				elif 'tokenArray' in words[-1]:
					return

				i += 1
				if len(words) == 0:
					return
				else:
					prevObject = words[-1]

		else:
			raise MyException('check this out')




class Quote(dict):


	def __init__(self, *args, **kwargs):
		super(Quote, self).__init__(*args, **kwargs)
		mandatory_listy_attributes = [
			'tokenArray', 'attribution_id', 'token:start', 
			'token:stop', 'sentence_id', 'parag_id']

		for attr in mandatory_listy_attributes:
			if attr not in self:
				self[attr] = None


	def truncated(self):
		'''
			return a simple single-line string made from all the tokens in 
			the quote.  This is the way the quote actually appears in the text
		'''
		# note, the first token is a "root token", which has to be skipped
		return '<QUOTE>'


	def __str__(self):

		offset = '(%d,%d)' % ( 
			self['token:start'], 
			self['token:stop']
		)

		tokens = ' '.join([t['word'] for t in self['tokenArray']])


		description = '%s: %s %s %s' % (
			'<QUOTE>', self['attribution_id'], offset, tokens
		)

		description = description.encode('utf8')

		return description


	def __repr__(self):
		return self.__str__()


	def setTokenList(tokenArrayLIST):
		for token in tokenArrayLIST:
			if 'pos' in token:
				self['tokenArray'] = tokenArrayLIST
			else:
				raise MyException("you tried declaring a quote with a list of objects that are not tokens")

	def addTokenToList(token):
		if 'pos' in token:
			self['tokenArray'].append(token)


class SourceSpeakerEntity(dict):


	def __init__(self, *args, **kwargs):
		super(SourceSpeakerEntity, self).__init__(*args, **kwargs)
		mandatory_listy_attributes = [
			'tokenArray', 'attribution_id', 'token:start', 
			'token:stop', 'sentence_id', 'parag_id', 'entity_idx']

		self.appearance = '<ENTITY>'

		for attr in mandatory_listy_attributes:
			if attr not in self:
				self[attr] = None



	def truncated(self):
		'''
			return a simple single-line string made from all the tokens in 
			the quote.  This is the way the quote actually appears in the text
		'''
		# note, the first token is a "root token", which has to be skipped
		return '<ENTITY>'


	def __str__(self):
		
		offset = '(%d,%d)' % ( 
			self['token:start'], 
			self['token:stop']
		)

		tokens = ' '.join([t['word'] for t in self['tokenArray']])


		description = '%s: %s %s %s' % (
			'<ENTITY>', self['attribution_id'], offset, tokens
		)

		description = description.encode('utf8')

		return description


	def __repr__(self):
		return self.__str__()


	def setTokenList(self, tokenArrayLIST):
		for token in tokenArrayLIST:
			if 'pos' in token:
				self['tokenArray'] = tokenArrayLIST
			else:
				raise MyException("you tried declaring an entity with a list of objects that are not tokens")

	def addTokenToList(self, token):
		if 'pos' in token:
			if self['tokenArray'] != None:
				self['tokenArray'].append(token)
			else:
				self['tokenArray'] = [token]
		else:
			raise MyException("you tried declaring an entity with a list of objects that are not tokens")



class reportingVerb(dict):

	def __init__(self, *args, **kwargs):
		super(reportingVerb, self).__init__(*args, **kwargs)
		mandatory_listy_attributes = [
			'tokenArray', 'attribution_id', 'token:start', 
			'token:stop', 'sentence_id', 'parag_id']

		for attr in mandatory_listy_attributes:
			if attr not in self:
				self[attr] = None




	def truncated(self):
		'''
			return a simple single-line string made from all the tokens in 
			the quote.  This is the way the quote actually appears in the text
		'''
		# note, the first token is a "root token", which has to be skipped
		return '<VERB>'


	def __str__(self):

		offset = '(%d,%d)' % ( 
			self['token:start'], 
			self['token:stop']
		)

		tokens = ' '.join([t['word'] for t in self['tokenArray']])


		description = '%s: %s %s %s' % (
			'<VERB>', self['attribution_id'], offset, tokens
		)

		description = description.encode('utf8')

		return description


	def __repr__(self):
		return self.__str__()


	def setTokenList(tokenArrayLIST):
		for token in tokenArrayLIST:
			if 'pos' in token:
				self['tokenArray'] = tokenArrayLIST
			else:
				raise MyException("you tried declaring an entity with a list of objects that are not tokens")

	def addTokenToList(token):
		if 'pos' in token:
			if self['tokenArray'] != None:
				self['tokenArray'].append(token)
			else:
				self['tokenArray'] = [token]
		else:
			raise MyException("you tried declaring an entity with a list of objects that are not tokens")




class MyException(Exception):
    pass
	



if __name__ == '__main__':
   main()

