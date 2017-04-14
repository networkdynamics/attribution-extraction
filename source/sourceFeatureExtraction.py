import sourceEntityTrainingSet as tokenize
import resolveSourceSpan as sourceSpanning
import resolveCue as cueResolving
import sys
import os
#from intermediaries.nlpReaders.annotated_text import AnnotatedText as A
#from intermediaries.nlpReaders.parc_reader import AnnotatedText as B
from parc_reader import ParcCorenlpReader as P
import csv
import itertools
import operator
import string
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from collections import defaultdict
from sklearn.externals import joblib
import multiprocessing
from multiprocessing import Manager
import pdb
import time
from SETTINGS import PACKAGE_DIR

titles = ['executive', 'chairman', 'official', 'spokesman', 'spokeswoman', 'officer']
numParag = 9

SOURCE_TRAINER_PATH = os.path.join(PACKAGE_DIR, 'sourceTrainer2.pkl')


#fixes all the omnipresent unicode issues
print sys.getdefaultencoding()
reload(sys)
sys.setdefaultencoding('utf-8')

#change this if you would only like to do a certain number of files, useful for testing
maxNumFiles = -1

#base dir for all data files
data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/'))

def workerFunction(myFile, coreNlPFiles, annotatedFiles, rawFiles, predictedQuotes, return_list):
	#in case you want a minimum file number
#	if 'wsj_1567' not in myFile:
#		return return_list

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
			return return_list

	myRawFile = [s for s in rawFiles if fileNoXML in s][0]
	

	#open the file, extract the features and return all the rows
	try:
		if annotatedFiles != None:
			try:
				fileRows, article = tokenize.openFile(myFile, myAnnotatedFile, myRawFile, None)
				theseTokens = postprocess(fileRows, article, True)
				for perm in theseTokens:
					row, headAttrs = turnToRow(perm, True)
					print row
					print headAttrs
					return_list.append(row)
				print len(theseTokens)
				return return_list
			except:
				raise
				return return_list
				

		if annotatedFiles == None:
			fileRows, article = tokenize.openFile(myFile, None, myRawFile, predictedQuotes)
			theseTokens = postprocess(fileRows, article, False)
			article = sourceSpanning.resolveSourceSpan(article, fileRows, theseTokens)
			print 'here'
			cueResolving.linkCue(article)
			
	except:
		raise
		return return_list


		#myRows += theseTokens


		#numTokens = len(myRows)

def chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in xrange(0, len(l), n))


def findFiles(coreNlPFiles, annotatedFiles, rawFiles, predictedQuotes, output):

	#open each NLP File
	j = 0
	splitLists = list(chunks(coreNlPFiles, len(coreNlPFiles)/15))


	lastList = splitLists[-1]
	del splitLists[-1]

	lengthLists = len(splitLists[0])

	jobs = []
	manager = Manager()
	return_list = manager.list()

	#first lists are all equally sized, pick one from each at each iteration
	for i in range(lengthLists):
		for thisList in splitLists:
			myFile = thisList[i]
			p = multiprocessing.Process(target = workerFunction, args=(myFile, coreNlPFiles, annotatedFiles, rawFiles, predictedQuotes, return_list))
			jobs.append(p)
			p.start()
			#break
		time.sleep(5)
		#break

	#append the files from last list (remainder of total files divided by 10)
	
	print len(return_list)

	for myFile in lastList:
		p = multiprocessing.Process(target = workerFunction, args=(myFile, coreNlPFiles, annotatedFiles, rawFiles, predictedQuotes, return_list))
		jobs.append(p)
		p.start()

	for proc in jobs:
		proc.join()

	print 'printing to file'

	print len(return_list)
	if annotatedFiles != None:
		writeToCSV(return_list, output)

	return return_list

def findFiles1(coreNlPFiles, annotatedFiles, rawFiles, predictedQuotes, output):
	j = 0
	myRows = []
	return_list = []
	for myFile in coreNlPFiles:
		if j < -1:
			j = j + 1
			continue
		else:
			return_list = workerFunction(myFile, coreNlPFiles, annotatedFiles, rawFiles, predictedQuotes, return_list)


	return myRows

def findVerb(sentence, indxQuote):
	verbs = []
	tokens = sentence['words']
	for indx, token in enumerate(tokens):
		if 'tokenArray' in token:
			if token.truncated() == '<VERB>':
				verbs.append((token, abs(indxQuote - indx)))

	#print verbs
	if verbs == []:
		return None, None

	verb = min(verbs, key=operator.itemgetter(1))[0]

	#print verb.keys()
	tokens = verb['tokenArray']

	for token in tokens:
		#print token.keys()
	
		for child in token['children']:
			if child[0] == 'nsubj':
				return verb, child[1]	
		for parent in token['parents']:
			if parent[0] == 'nsubj':
				return verb, parent[1]

		return verb, None	




def postprocess(fileRows, article, flagLabelled):
	listEntities = []
	listQuotes = []

	for sentence in fileRows:
		tokens = sentence['words']
		for token in tokens:
			if 'tokenArray' in token:
				if token.truncated() == '<QUOTE>':
					listQuotes.append(token)
				elif token.truncated() == '<ENTITY>':
					listEntities.append(token)


	perms = []


	for entity, quote in itertools.product(listEntities,listQuotes):
		#if quote['parag_id'] - entity['parag_id'] < numParag and quote['parag_id'] >= entity['parag_id']:
		if quote['parag_id'] == entity['parag_id']:
			perms.append((entity, quote))

	flatFile = [y for x in fileRows for y in x['words']]

	permsSoFar = []
	allPerms = []

	quoteSet = list(set([perm[1]['attribution_id'] for perm in perms]))

	if flagLabelled == True:
		
		for quote in quoteSet:
			thesePerms = []
			for perm in perms:
				if perm[1]['attribution_id'] == quote:
					thesePerms.append(perm)

			if len(thesePerms) >= 70:
				continue

			correct, thesePerms, fileRows = assignLabelAnnotated(thesePerms, fileRows, article)
			if correct == None:
				continue
			flatFile = [y for x in fileRows for y in x['words']]

			found = False
			for perm in thesePerms:
				if correct == perm:
					allPerms.append((perm[0], perm[1], 'Y'))
					found = True
				else:
					allPerms.append((perm[0], perm[1], 'N'))

			if found == False:
				print 'FALSE FALSE FALSE'
				for perm in thesePerms:
					attrID = perm[1]['attribution_id']
					print article.attributions[attrID]

		for perm in allPerms:
			thisPerm = featureExtract(perm, flatFile, fileRows, permsSoFar, flagLabelled, article)
			if thisPerm != None:
				permsSoFar.append(thisPerm)
			else:
				continue

		return permsSoFar

	for quote in quoteSet:
		thesePerms = []
		entityFeats = []
		for perm in perms:
			if perm[1]['attribution_id'] == quote:
				thesePerms.append((perm[0], perm[1], None))

		if len(thesePerms) >= 70:
				continue

		for perm in thesePerms:
			thisPerm = featureExtract(perm, flatFile, fileRows, permsSoFar, flagLabelled, article)
			entityFeats.append(thisPerm)

		maxval = liblinear(entityFeats)

		for indx, perm in enumerate(entityFeats):
			if maxval == None:
				perm.set_label('N')
			if indx == maxval:
				perm.set_label('Y')
			else:
				perm.set_label('N')

			permsSoFar.append(perm)

	return permsSoFar


def assignLabelAnnotated(quoteSet, fileRows, article):
	goldAttrId = quoteSet[0][1]['attribution_id']

	goldAttr = article.attributions[goldAttrId]
	goldSource = goldAttr['source']
	#headOfGoldSource = article._find_head(goldSource)
	if goldSource == []:
		return None, quoteSet, fileRows
	goldSourceStartId = goldSource[0]['id']
	goldSourceStopId = goldSource[-1]['id']
	goldSourceSentId = goldSource[0]['sentence_id']

	permsInGoldSource = []

	for perm in quoteSet:
		currentEntity = perm[0]
		for token in goldSource:
			for entityToken in currentEntity['tokenArray']:
				if entityToken['id'] == token['id'] and entityToken['sentence_id'] == token['sentence_id']:
					permsInGoldSource.append(perm)
	
	if len(permsInGoldSource) == 1:
		return permsInGoldSource[0], quoteSet, fileRows
	elif len(permsInGoldSource) > 1:
		for perm in permsInGoldSource:
			entity = perm[0]
			#for token in entity['tokenArray']:
				#for head in headOfGoldSource:
				#	if token['id'] == head['id'] and token['sentence_id'] == head['sentence_id']:
				#		return perm, quoteSet, fileRows
			for token in entity['tokenArray']:
				if token['ner'] == 'PERSON':
					return perm, quoteSet, fileRows
			for token in entity['tokenArray']:
				if token['ner'] == 'ORGANIZATION':
					return perm, quoteSet, fileRows

		#return first found
		return permsInGoldSource[0], quoteSet, fileRows
	

	#no identified entities found in the gold span, turn whole gold span into entity, change the fileRows
	elif len(permsInGoldSource) == 0:
		entity = SourceSpeakerEntity()
		entity['tokenArray'] = goldSource
		entity['attribution_id'] = goldAttrId
		entity['token:start'] = goldSource[0]['id']
		entity['token:stop'] = goldSource[-1]['id']
		entity['sentence_id'] = goldSourceSentId
		entity['parag_id'] = goldSource[0]['parag_id']
		newPerm = (entity, perm[1])
		quoteSet.append(newPerm)

		for row in fileRows:
			if row['sentence_id'] == goldSourceSentId and row['parag_id'] == goldSource[0]['parag_id']:
				print row
				print article.sentences[goldSourceSentId]
				index = fileRows.index(row)
				newTokenArray = []
				firstToken = goldSource[0]['id']
				lastToken = goldSource[-1]['id']
				added = False

				for token in row['words']:
					if 'pos' in token and (token['id'] < firstToken or token['id'] > lastToken):
						newTokenArray.append(token)
					elif 'pos' in token:
						if added == False:
							newTokenArray.append(entity)
							added = True
						else:
							continue
					elif token['token:start'] >= firstToken:
						if added == False:
							newTokenArray.append(entity)
							newTokenArray.append(token)
							added = True
						else:
							newTokenArray.append(token)

					elif token['token:start'] <= firstToken and token['token:stop'] >= lastToken:
						newTokenArray.append(token)
						newTokenArray.append(entity)
						entity.ent
					else:
						newTokenArray.append(token)

				row['words'] = newTokenArray
				fileRows[index] = row

		return newPerm, quoteSet, fileRows


	print goldSource
	print fileRows
	print headOfGoldSource
	entity.end
	print 'DIDNT FIND ANYTHING'


classifier = joblib.load(SOURCE_TRAINER_PATH) 

def liblinear(featsArray):

	featureArray = []
	for thisPerm in featsArray:
		row, head = turnToRow(thisPerm, False)
		split = row.split(',')
		split = [float(val) for val in split]
		split = np.array(split)
		featureArray.append(split)



	arrayProbas = classifier.predict_proba(featureArray)
	positics = np.delete(arrayProbas, 0, 1)

	maxval = np.argmax(positics)
	if positics[maxval][0] < 0.05:
		return None
	return maxval

def writeToCSV(allPerms, output):

	attributes = [i for i in dir(allPerms[0]) if not i.startswith('__')]
	attributes = [i for i in attributes if not i.startswith('set_')]


	with open(os.path.join(data_dir, output), 'wb') as myfile:
		for permutation in allPerms:
			#row, headAttrs = turnToRow(permutation, True)
			myfile.write(permutation + '\n')
	myfile.close()

def turnToRow(perm, flagLabelled):
	attributes = [i for i in dir(perm) if not i.startswith('__')]
	attributes = [i for i in attributes if not i.startswith('set_')]

	row = ''

	headAttrs = ''

	for attr in attributes:
		if flagLabelled == True:

			attribute = getattr(perm, attr)

			if attr == 'quoteObject':
				metadata = attribute['attribution_id']
				row = metadata + ',' + row
				headAttrs = 'metadata,' + headAttrs

				continue
			elif attr == 'entityObject':
				continue
			elif attr == 'label':
				row = attribute + ',' + row
				headAttrs = 'label' + headAttrs
			else:
				if type(attribute) == int:
					row = row + str(attribute) + ','
					headAttrs = headAttrs + ',' + attr
				elif attribute == True:
					row = row + '1,'
					headAttrs = headAttrs + ',' + attr
				elif attribute == False:
					row = row + '0,'
					headAttrs = headAttrs + ',' + attr
				else:
					print attr
					print attribute
		else:
			attribute = getattr(perm, attr)

			if attr == 'quoteObject':
				continue
			elif attr == 'entityObject':
				continue
			elif attr == 'label':
				continue
			else:
				if type(attribute) == int:
					row = row + str(attribute) + ','
					headAttrs = headAttrs + attr + ','
				elif attribute == True:
					row = row + '1,'
					headAttrs = headAttrs + attr + ','
				elif attribute == False:
					row = row + '0,'
					headAttrs = headAttrs + attr + ','
				else:
					print attr
					print attribute


	return row[:-1], headAttrs



def featureExtract(perm, flatFile, fileRows, permsSoFar, flagLabelled, article):
	(entity, quote, label) = perm

	thisPerm = Permutation(entity, quote)

	entityAttr = entity['attribution_id']
	quoteAttr = quote['attribution_id']

	if flagLabelled == True:
		thisPerm.set_label(label)


	#set all distance features
	thisPerm = addDistanceFeats(perm, flatFile, thisPerm)
	thisPerm = addParagFeats(perm, flatFile, fileRows, thisPerm, article)
	thisPerm = addNearbyFeats(perm, flatFile, fileRows, thisPerm)
	
	thisPerm = addQuotationFeatures(perm, flatFile, fileRows, thisPerm, permsSoFar)
	thisPerm = addSequenceFeatures(perm, permsSoFar, flatFile, fileRows, thisPerm)

	return thisPerm


def addSequenceFeatures(perm, permsSoFar, flatFile, fileRows, thisPerm):


	(entity, quote, label) = perm

	correctPerms = []

	for aPastPerm in permsSoFar:
		if quote['parag_id'] <= aPastPerm.quoteObject['parag_id'] + numParag:
			if getattr(aPastPerm, 'label') == 'Y':
				correctPerms.append(aPastPerm)



	listPastSpeakers = []
	for myPerm in correctPerms:
		listPastSpeakers.append(myPerm.entityObject['tokenArray'])


	containsMentions = False
	coreferenceTokens = []

	for token in thisPerm.entityObject['tokenArray']:

		if 'mentions' in token:
			mention = token['mentions']
			for elem in mention:
				ref = elem['reference']
				for mention in ref['mentions']:
					found = False
					for coref in coreferenceTokens:
						firstTok = coref[0]
						lastTok = coref[-1]
						firstMen = mention['tokens'][0]
						lastMen = mention['tokens'][-1]
						if firstMen['id'] == firstTok['id'] and lastMen['sentence_id'] ==lastMen['sentence_id']:
							found = True
							break
					if found == False:
						coreferenceTokens.append(mention['tokens'])


	flatCoreference = [item for sublist in coreferenceTokens for item in sublist]

	timesSpeakerMentioned = 0

	for speaker in listPastSpeakers:
		token = speaker[0]['character_offset_begin']
		for coreference in coreferenceTokens:
			if token == coreference[0]['character_offset_begin']:
				timesSpeakerMentioned += 1
				break

	otherSpeakers = len(listPastSpeakers) - timesSpeakerMentioned

	entityArray = thisPerm.entityObject['tokenArray']
	headTokens = find_head(entityArray)

	isNsubj = False
	isNsubjVerbCue = False


	verbs = []
	for token in flatFile:
		if 'tokenArray' in token:
			if token.truncated() == '<VERB>':
				for eachToken in token['tokenArray']:
					verbs.append((eachToken['id'], eachToken['sentence_id']))

	for token in headTokens:
		parents = token['parents']
		for parent in parents:
			relation = parent[0]
			if 'nsubj' in relation:
				isNsubj = True
				parentToken = parent[1]
				if (parentToken['id'], parentToken['sentence_id']) in verbs:
					isNsubjVerbCue = True
				break
		if isNsubj == True:
			break



	thisPerm.set_sequenceFeats(timesSpeakerMentioned, otherSpeakers, isNsubj, isNsubjVerbCue)

	return thisPerm


#Distance features: number of words/paragraphs/quotations/entity mentions be- tween Q and S.
def addDistanceFeats(perm, flatFile, thisPerm):
	(entity, quote, label) = perm

	entitySent = -1
	entityPosition = -1

	quoteSent = -1
	quotePosition = -1

	entityAttr = entity['attribution_id']
	quoteAttr = quote['attribution_id']
	
	entityPosition = flatFile.index(entity)
	quotePosition = flatFile.index(quote)

	familiarSequence = False

	try:
		if flatFile[entityPosition + 1].truncated() == '<VERB>' and entityPosition + 2 == quotePosition:
			familiarSequence = True
	except:
		pass

	try:
		if flatFile[quotePosition + 1].truncated() == '<VERB>' and quotePosition + 2 == entityPosition:
			familiarSequence = True

	except:
		pass

	first = min(entityPosition, quotePosition)
	second = max(entityPosition, quotePosition)

	found = False
	countQuotes = 0
	countEntities = 0

	for index, elem in enumerate(flatFile):
		if index == first:
			found = True
		elif index == second:
			break
		elif 'pos' in elem:
			continue
		elif found == True and elem.truncated() == '<QUOTE>':
			countQuotes += 1
		elif found == True and elem.truncated() == '<ENTITY>':
			countEntities += 1


	wordFeatDist = abs(entityPosition - quotePosition)
	paragFeatDist = abs(entity['parag_id'] - quote['parag_id'])
	quoteFeatDist = countQuotes
	entityFeatDist = countEntities

	thisPerm.set_distanceFeats(wordFeatDist, paragFeatDist, quoteFeatDist, entityFeatDist, familiarSequence)

	return thisPerm

#Paragraph features: number of times S is mentioned and number of words and quotations in the paragraph including the quote and in the preceding 9 para- graphs.
def addParagFeats(perm, flatFile, fileRows, thisPerm, article):
	#mentions: ['head', 'end', 'reference', 'tokens', 'start', 'sentence_id']
	(entity, quote, label) = perm


	entityTokens = entity['tokenArray']
	sent_id = entity['sentence_id']
	parag_id = entity['parag_id']

	quoteParag_id = quote['parag_id']

	thisParWordCount = 0
	pastParsWordCount = 0
	quoteThisParag = 0
	quotePastParag = 0
	speakerReferences = 0

	#count number of quotes and words in paragraph including quote and in preceding 9 paragraphs
	for token in flatFile:
		if token['parag_id'] > quoteParag_id:
			break
		elif token['parag_id'] == quoteParag_id:
			thisParWordCount += 1
			if 'pos' not in token and token.truncated() == '<QUOTE>':
				quoteThisParag += 1
		elif token['parag_id'] > quoteParag_id - numParag:
			pastParsWordCount += 1
			if 'pos' not in token and token.truncated() == '<QUOTE>':
				quotePastParag += 1
		else:
			continue

	numMentionsThisPar = 0
	numMentionsPrevPar = 0
	#take head of entity and check for mentions in paragraph and previous 9 paragraphs
	head = article._find_head(entityTokens)


	listMentions = []
	if len(head) == 1:
		if 'mentions' not in head:
			head[0]['mentions'] = []
		mentions = head[0]['mentions']
		for mention in mentions:
			mentionHead = mention['head']
			references = mention['reference']['mentions']
			for reference in references:
				listMentions.append(reference['tokens'])
	else:
		currentLongest = []
		for headToken in head:
			listMentions = []
			if 'mentions' not in headToken:
				headToken['mentions'] = []
			mentions = headToken['mentions']
			for mention in mentions:
				mentionHead = mention['head']
				references = mention['reference']['mentions']
				for reference in references:
					listMentions.append(reference['tokens'])
				if len(listMentions) > len(currentLongest):
					currentLongest = listMentions
		listMentions = currentLongest

	for mention in listMentions:
		if mention[0]['parag_id'] > quoteParag_id:
			continue
		elif mention[0]['parag_id'] == quoteParag_id:
			numMentionsThisPar += 1
		elif mention[0]['parag_id'] > quoteParag_id - numParag:
			numMentionsPrevPar += 1




	thisPerm.set_paragraphFeats(numMentionsThisPar, numMentionsPrevPar, thisParWordCount, quoteThisParag, pastParsWordCount, quotePastParag)
	return thisPerm

#Nearby features: whether the tokens to the right or left of Q and S are punctua- tion/ another speaker mention/ another quote/ an identified speech verb.

def addNearbyFeats(perm, flatFile, fileRows, thisPerm):
	(entity, quote, label) = perm

	#prev and next features entity
	entityPosition = flatFile.index(entity)
	entitySentID = entity['sentence_id']

	if entityPosition >= 1 and flatFile[entityPosition - 1]['sentence_id'] == entitySentID:
		prevToken = flatFile[entityPosition - 1]
		featPrevVerbE = checkToken('VERB', prevToken)
		featPrevQuoteE = checkToken('QUOTE', prevToken)
		featPrevSpeakerE = checkToken('SPEAKER', prevToken)
		featPrevPunctE = checkToken('PUNCT', prevToken)
	else:
		featPrevVerbE = False
		featPrevQuoteE = False
		featPrevSpeakerE = False
		featPrevPunctE = False

	if entityPosition < len(flatFile) - 1 and flatFile[entityPosition + 1]['sentence_id'] == entitySentID:
		nextToken = flatFile[entityPosition + 1]
		featNextVerbE = checkToken('VERB', nextToken)
		featNextQuoteE = checkToken('QUOTE', nextToken)
		featNextSpeakerE = checkToken('SPEAKER', nextToken)
		featNextPunctE = checkToken('PUNCT', nextToken)
	else:
		featNextVerbE = False
		featNextQuoteE = False
		featNextSpeakerE = False
		featNextPunctE = False

	#prev and next features quote
	quotePosition = flatFile.index(quote)
	quoteSentID = quote['sentence_id']

	if quotePosition >= 1 and flatFile[quotePosition - 1]['sentence_id'] == quoteSentID:
		prevToken = flatFile[quotePosition - 1]
		featPrevVerbQ = checkToken('VERB', prevToken)
		featPrevQuoteQ = checkToken('QUOTE', prevToken)
		featPrevSpeakerQ = checkToken('SPEAKER', prevToken)
		featPrevPunctQ = checkToken('PUNCT', prevToken)
	else:
		featPrevVerbQ = False
		featPrevQuoteQ = False
		featPrevSpeakerQ = False
		featPrevPunctQ = False

	if quotePosition < len(flatFile) - 1 and flatFile[quotePosition + 1]['sentence_id'] == quoteSentID:
		nextToken = flatFile[quotePosition + 1]
		featNextVerbQ = checkToken('VERB', nextToken)
		featNextQuoteQ = checkToken('QUOTE', nextToken)
		featNextSpeakerQ = checkToken('SPEAKER', nextToken)
		featNextPunctQ = checkToken('PUNCT', nextToken)
	else:
		featNextVerbQ = False
		featNextQuoteQ = False
		featNextSpeakerQ = False
		featNextPunctQ = False

	thisPerm.set_nearbyFeats((featPrevVerbE, featPrevQuoteE, featPrevSpeakerE, featPrevPunctE), 
		(featNextVerbE, featNextQuoteE, featNextSpeakerE, featNextPunctE),
		(featPrevVerbQ, featPrevQuoteQ, featPrevSpeakerQ, featPrevPunctQ), 
		(featNextVerbQ, featNextQuoteQ, featNextSpeakerQ, featNextPunctQ))	

	return thisPerm


def checkToken(selector, token):
	if selector == 'VERB':
		if 'tokenArray' in token:
			if token.truncated() == '<VERB>':
				return True
	elif selector == 'QUOTE':
		if 'tokenArray' in token:
			if token.truncated() == '<QUOTE>':
				return True
	elif selector == 'SPEAKER':
		if 'tokenArray' in token:
			if token.truncated() == '<ENTITY>':
				return True
	elif selector == 'PUNCT':
		if 'pos' in token:
			if token['pos'] in string.punctuation:
				return True

	return False

#quotation features: whether S or other speakers are mentioned in Q; Q distance from the beginning of the paragraph; word length of Q.

def addQuotationFeatures(perm, flatFile, fileRows, thisPerm, permsSoFar):
	(entity, quote, label) = perm
	parag_id = quote['parag_id']

	lengthQuote = len(quote['tokenArray'])


	count = -1
	count = 0
	for elem in flatFile:
		if elem['parag_id'] == parag_id:
			try:
				count = flatFile.index(elem)
				break
			except:
				break
		count = count + 1

	distance = flatFile.index(quote) - count

	permsEntityQuote = []
	for perm in permsSoFar:
		permsEntityQuote.append((perm.entityObject, perm.quoteObject))


	flatQuoteTokens = ' '.join([word['word'] for word in quote['tokenArray']])
	stringPerms = list(set([' '.join(word['word'] for word in entity['tokenArray']) for (entity, quote) in permsEntityQuote]))
	flatEnityTokens = ' '.join([word['word'] for word in entity['tokenArray']])

	booleanThisEntity = False
	booleanAnotherEntity = False

	booleanAnotherEntity = any((substring + ' ') in flatQuoteTokens for substring in stringPerms)

	booleanThisEntity = (flatEnityTokens + ' ') in flatQuoteTokens


	thisPerm.set_quotationFeats(booleanThisEntity, booleanAnotherEntity, distance, lengthQuote)

	return thisPerm


def main():
	#pdb.set_trace()
	usageMessage = '\nCorrect usage of the Entity Source Feature Extractor command is as follows: \n' + \
					'\n\n WHEN AN ANNOTATED FILESET EXISTS TO GET LABELS FROM:\n' + \
					'To extract tokens and their features: \n python source/createTrainSet.py -labelled /pathToCoreNLPDirectory /pathToAnnotatedFilesDirectory /pathToRawDirectory \n' + \
					'\n\n WHEN THE LABELS ARE UNKNOWN:\n' + \
					'To extract tokens and their features: \n python source/createTrainSet.py -unlabelled /pathToCoreNLPDirectory /pathToRawDirectory /pathToPredictedQuotes \n' + \
					'\nFor reference, the path to the CoreNLP file is: /home/ndg/dataset/ptb2-corenlp/CoreNLP/ + train, test or dev depending on your needs. \n' + \
					'The path to the Parc3 files is /home/ndg/dataset/parc3/ + train, test or dev depending on your needs.\n'

	args = sys.argv

	if len(args) == 6:

		flag = args[1]
		

		if flag == '-labelled':
			pathToCORENLP = args[2]
			pathToAnnotatedFiles = args[3] 
			pathToRaw = args[4]
			output = args[5]

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


			coreNLPFiles = tokenize.openDirectory(pathToCORENLP)
			annotatedFiles = tokenize.openDirectory(pathToAnnotatedFiles)
			rawFiles = tokenize.openDirectory(pathToRaw)


			findFiles(coreNLPFiles, annotatedFiles, rawFiles, None, output)

		elif flag == '-unlabelled':
			flag = args[1]
			pathToCORENLP = args[2]
			pathToRaw = args[3]
			pathToPredictedQuotes = args[4]
			output = None

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


			with open(os.path.join(data_dir, pathToPredictedQuotes), 'r') as f:
				listQuotes = []
				listQuotes = f.readlines()

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
				

			coreNLPFiles = tokenize.openDirectory(pathToCORENLP)
			rawFiles = tokenize.openDirectory(pathToRaw)

			findFiles(coreNLPFiles, None, rawFiles, quoteDict, output)

		else:
			print usageMessage
			return

class Permutation(object):

	#Features come from Pareti_Attribution_Dissertation.pdf page 145

	quoteObject = None
	entityObject = None
	#a match or not
	label = None
	#added
	containsPerson = None
	#containsOrg = None
	sameSentence = None
	#familiarSequence = None


	#Distance Features
	numWordsBetween = None
	#numParagBetween = None
	numQuotesBetween = None
	numEntityBetween = None
	#added
	isNSubj = None
	isNSubjVC = None

	#Paragraph Features
	numMentionsThisPar = None
	numMentionsPrevPar = None
	numWordsParag = None
	numQuotesParag = None
	numWordsPrev9Parag = None
	numQuotesPrev9Parag = None

	#Nearby Features
	featNextVerbQ = None
	#featNextQuoteQ = None
	featNextSpeakerQ = None
	featNextPunctQ = None

	featPrevVerbQ = None
	#featPrevQuoteQ = None
	featPrevSpeakerQ = None
	featPrevPunctQ = None

	featNextVerbE = None
	featNextQuoteE = None
	featNextSpeakerE = None
	featNextPunctE = None

	featPrevVerbE = None
	featPrevQuoteE = None
	featPrevSpeakerE = None
	featPrevPunctE = None

	#Quotation Features
	thisSpeakerMentioned = None
	otherSpeakerMentioned = None
	quoteDistanceFromParag = None
	wordLengthQuote = None

	#Sequence Features
	numQuotesAttributed = None
	numQuotesOtherSpeakers = None


	def __init__(self, Entity, Quote):
		self.quoteObject = Quote
		self.entityObject = Entity

		
		if Quote['sentence_id'] == Entity['sentence_id']:
			self.sameSentence = True
		else:
			self.sameSentence = False

		for token in Entity['tokenArray']:
			if token['ner'] == 'PERSON':
				self.containsPerson = True
			#	self.containsOrg = False
				break
			#if token['ner'] == 'ORGANIZATION':
			#	self.containsOrg = True
			#	self.containsPerson = False
			#	break

		#if self.containsOrg == None:
		#	self.containsOrg = False

		if self.containsPerson == None:
			self.containsPerson = False
		


	def set_distanceFeats(self, words, parags, quotes, entities, familiarSequence):
		self.numWordsBetween = words
		#self.numParagBetween = parags
		self.numQuotesBetween = quotes
		self.numEntityBetween = entities
		#self.familiarSequence = familiarSequence

	def set_paragraphFeats(self, mentions, numMentionsPrevPara, paragWords, paragQuotes, prevParagWords, prevParagQuotes):

		self.numMentionsThisPar = mentions
		self.numMentionsPrevPar = numMentionsPrevPara
		self.numWordsParag = paragWords

		self.numQuotesParag = paragQuotes
		self.numWordsPrev9Parag = prevParagWords
		self.numQuotesPrev9Parag = prevParagQuotes

	def set_nearbyFeats(self, entityPrev, entityNext, quotePrev, quoteNext):
		
		(self.featNextVerbQ, hi, self.featNextSpeakerQ, self.featNextPunctQ) = quoteNext
		(self.featPrevVerbQ, hi, self.featPrevSpeakerQ, self.featPrevPunctQ) = quotePrev
		(self.featNextVerbE, self.featNextQuoteE, self.featNextSpeakerE, self.featNextPunctE) = entityNext
		(self.featPrevVerbE, self.featPrevQuoteE, self.featPrevSpeakerE, self.featPrevPunctE) = entityPrev



	def set_quotationFeats(self, thisSpeaker, otherSpeaker, distance, length):
		self.thisSpeakerMentioned = thisSpeaker
		self.otherSpeakerMentioned = otherSpeaker
		self.quoteDistanceFromParag = distance
		self.wordLengthQuote = length

	def set_sequenceFeats(self, numThisSpeaker, numOtherSpeakers, nsubj, NSubjVC):
		self.numQuotesAttributed = numThisSpeaker
		self.numQuotesOtherSpeakers = numOtherSpeakers
		self.isNSubj = nsubj
		self.isNSubjVC = NSubjVC



	def set_label(self, value):
		self.label = value


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
			self['words'] = []

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
					if len(words) == 1:
						del words[0]
				elif 'pos' in prevObject: 
					return
				elif prevObject.truncated() == '<ENTITY>':
					prevTokenArray = words[-1]['tokenArray']
					prevTokens = [word['word'] for word in prevTokenArray]
					theseTokens = [word['word'] for word in nominalTokens]

					if set(prevTokens) < set(theseTokens):
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

def find_head(tokens):

		heads = []

		# If there is only one token, that's the head
		if len(tokens) ==  1:
			heads = [tokens[0]]

		else:

			# otherwise iterate over all the tokens to find the head
			for token in tokens:

				# if this token has no parents or children its not part
				# of the dependency tree (it's a preposition, e.g.)
				if 'parents' not in token and 'children' not in token:
					continue

				# if this token has any parents that among the tokens list
				# it's not the head!
				try:

					token_ids = [
						(t['sentence_id'], t['id']) for t in tokens
					]

					has_parent_in_span = any([
						(t[1]['sentence_id'], t[1]['id'])
						in token_ids for t in token['parents']
					])

					if has_parent_in_span:
						relations_to_parents = []
						for t in token['parents']:
							for eachToken in tokens:
								if t[1]['id'] == eachToken['id'] and t[1]['sentence_id'] == eachToken['sentence_id']:
									relations_to_parents.append(t)

						continue
				except KeyError:
					pass

				# otherwise it is the head
				else:
					heads.append(token)

		# NOTE: head may be none
		return heads


if __name__ == '__main__':
   main()


