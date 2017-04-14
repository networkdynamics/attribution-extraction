#this program runs the pipeline and then compares the predicted results with gold spans 

print 'importing'
import contentSpanExtractor as contentSpans
import intermediaries.kNN as kNN
import intermediaries.newkNN as newkNN
import sourceFeatureExtraction as resolveSource
import sourceEntityTrainingSet as tokenize
import resolveSourceSpan as sourceSpanning
import resolveCue as cueResolving
import crfsuite as crf
from parc_reader import ParcCorenlpReader as P
import sys
import os
import csv
import pycrfsuite
import copy
import verbCuesFeatureExtractor as verbCues
import multiprocessing
from multiprocessing import Manager
import pdb
from operator import itemgetter
from itertools import groupby


data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../', 'data/'))
xml_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../', 'data/xmlPredictions/'))

maxNumFiles = -1
#CuetrainingFile = os.path.join(data_dir, 'VerbFeatsForKNN.tsv')
CuetrainingFile = os.path.join(data_dir, 'PARCTrainVerbFeats.csv')

CuetrainingData = kNN.readData(CuetrainingFile, False)
tagger = pycrfsuite.Tagger()
tagger.open('ContentSpanClassifier3.crfsuite')

with open(data_dir + '/nounCues.csv', 'rb') as f:
    reader = csv.reader(f)
    nounCues = list(reader)

countPredicted = 0;
countGold = 0;


partialsPTotal = (0,0,0)
strictPTotal = (0,0,0)
softPTotal = (0,0,0)

partialsRTotal = (0,0,0)
strictRTotal = (0,0,0)
softRTotal = (0,0,0)

#open all files, run pipeline, check metrics
def label(path, pathToRaw, pathToAnnotated, outputFile):
	#open files
	listOfNLPFiles = verbCues.openDirectory(path)
	listOfRawFiles = verbCues.openDirectory(pathToRaw)
	listOfAnnotatedFiles = verbCues.openDirectory(pathToAnnotated)

	myRows = []
	j = 0

	#[strict, partialPrecision, partialRecall, partialFscore, soft]
	
	countPredicted = 0
	countGold = 0

	allAttrs = 0

	partialsPTotal = (0,0,0)
	strictPTotal = (0,0,0)
	softPTotal = (0,0,0)
	
	partialsRTotal = (0,0,0)
	strictRTotal = (0,0,0)
	softRTotal = (0,0,0)

	countDiscontinous = 0
	countRepeat = 0


	for myFile in listOfNLPFiles:

		if j < -1:
			j = j + 1
			continue

	###if i'm trying to find a specific file

		#if 'wsj_2370' not in myFile:
		#	continue


		files = len(listOfNLPFiles)
		filename = myFile.split('/')[-1]
		typeFile = myFile.split('/')[-2]
		fileNoXML = filename.split('.xml')[0]

		print filename		

		print('opening file: ' + myFile + ' ' + str(j) + ' out of ' + str(files))

	###resolve to raw and annotated files
		myRawFile = [s for s in listOfRawFiles if fileNoXML in s][0]
		myAnnotatedFile = [s for s in listOfAnnotatedFiles if filename in s]

		if len(myAnnotatedFile) == 1:
			myAnnotatedFile = myAnnotatedFile[0]
		else:
			continue

	###open real article with real attrs
		goldArticle = openFile(myFile, myAnnotatedFile, myRawFile)
		allAttrs += len(goldArticle.attributions)
		sentencesToSkip = []

		for attr in goldArticle.attributions:
			thisAttr = goldArticle.attributions[attr]
			content = thisAttr['content']
			source = thisAttr['source']
			cue  = thisAttr['cue']
			attrID = thisAttr['id']
			listIds = []
			if len(content) <= 1:
				goldArticle.remove_attribution(attr)
				continue
			if len(source) == 0:
				goldArticle.remove_attribution(attr)
				print 'removing'
				print thisAttr
				continue
			
			firstChar = content[0]['character_offset_begin']
			lastChar = content[-1]['character_offset_end']
			sourceFirstChar = source[0]['character_offset_begin']
			
			if sourceFirstChar > firstChar and sourceFirstChar < lastChar:
				goldArticle.remove_attribution(attr)
				
				
				ids = [token['id'] for token in content]
				ranges = []
				for k, g in groupby(enumerate(ids), lambda (i,x):i-x):
					group = map(itemgetter(0), g)
					ranges.append((group[0], group[-1]))
				lengths = [b - a for (a,b) in ranges]
				maxLength = max(lengths)


				newContents = []
				i = 0
				for thisRange in ranges:
					i += 1
					beginID = thisRange[0]
					endID = thisRange[1]
					lengthThisRange = endID - beginID
					if lengthThisRange == maxLength:
						newContent = content[beginID:endID]
						newContents.append(newContent)
						goldArticle.add_attribution(
								cue_tokens=cue,
								content_tokens=newContent,
								source_tokens=source,
								attribution_id=attrID + '_' + str(i),
						)
						break

				
				countDiscontinous = countDiscontinous + 1

				
				for token in content:
					sentenceID = token['sentence_id']
					if sentenceID not in sentencesToSkip:
						sentencesToSkip.append(sentenceID)
				
				print 'discontinous'
				print attr
				print content
			
			

		print sentencesToSkip
		
	###find verb cues and label with KNN
		thisFilesVerbs, article = verbCues.openFile(myFile, None, myRawFile)
		labelledVerbs = kNN.labelWithoutCSV(CuetrainingData, thisFilesVerbs)
		
		#labelledVerbs = newkNN.labelWithoutCSV(CuetrainingData, thisFilesVerbs)
		#print labelledVerbs
		#REVISIT
		
	###comment or uncomment to use predicted spans or gold spans and then tokenize the file into <quote> <entity> etc.

		#PREDICTED SPANS
		'''
		
		fileRows = contentSpans.findFeatures(filename, article, labelledVerbs, None)
		contentSpanLabels = operateCRFSuite(fileRows, outputFile)
		print contentSpanLabels
		tokenizedRows, newArticle = tokenize.process(article, filename, contentSpanLabels, labelledVerbs)
		
		'''
		
		#GOLD SPANS
		for attr in goldArticle.attributions:
			content = goldArticle.attributions[attr]['content']
			if len(content) < 1:
				goldArticle.remove_attribution(attr)
				continue
			newContent = []
			for token in content:
				sentID = token['sentence_id']
				tokID = token['id']
				newToken = article.sentences[sentID]['tokens'][tokID]
				newContent.append(newToken)

			article.add_attribution(
				cue_tokens=[],
				content_tokens= newContent,
				source_tokens = [],
				id_formatter = 'my_attribution_' + str(attr) + '_' + filename,
				)
		print goldArticle.attributions

		tokenizedRows, newArticle = tokenize.process(article, filename, [], labelledVerbs)
		
		
		#print tokenizedRows
		#print newArticle.sentences

		#'''
		
	###decide which entities are the best entity matches
		#REVISIT
		
		#'''
		theseTokens = resolveSource.postprocess(tokenizedRows, newArticle, False)



	###find the whole source span
		#REVISIT
		
		try:
			newArticle = sourceSpanning.resolveSourceSpan(newArticle, tokenizedRows, theseTokens)
			attrs = newArticle.attributions
			for attr in newArticle.attributions.keys():
				thisAttr = newArticle.attributions[attr]
				source = thisAttr['source']
				content = thisAttr['content']
				attrArticle = attr.split('wsj_')[1]
				attrArticle = attrArticle.split('.xml')[0]
				attrArticle = 'wsj_' + attrArticle
				if len(content) <= 1:
					print 'removing1'
					print attr
					remove_attribution1(newArticle, attr)
					continue
				if len(source) == 0:
					print 'removing1'
					print attr
					remove_attribution1(newArticle, attr)
					continue
				
				'''
				for token in content:
					sentenceID = token['sentence_id']
					if sentenceID in sentencesToSkip:
						print 'removing2'
						print attr
						remove_attribution1(newArticle, attr)
						break
				'''
						

			finalArticle = cueResolving.linkCue(newArticle, labelledVerbs, nounCues)
		except:
			raise


		#'''

		
		#writeToXml(typeFile, filename, finalArticle)
		
	###TESTING FINAL RESULTS#####
		#finalTests(article, finalArticle)
		#for attr in finalArticle.attributions:
		#	print finalArticle.attributions[attr]

		#for attr in goldArticle.attributions:
		#	print goldArticle.attributions[attr]



	###compare perfect article with predicted article and create a score
		
		#ps and rs scores
		

		print 'P'
		partialsPTotal, strictPTotal, softPTotal= overlap(finalArticle, goldArticle, partialsPTotal, strictPTotal, softPTotal, False)
		print 'R'
		partialsRTotal, strictRTotal, softRTotal = overlap(goldArticle, finalArticle, partialsRTotal, strictRTotal, softRTotal, True)

		'''
		articleSourceProportionsP, articleContentProportionsP, articleCueProportionsP = partialsPTotal
		articleSourceProportionsR, articleContentProportionsR, articleCueProportionsR = partialsRTotal

		articleSourceStrictP, articleContentStrictP, articleCueStrictP = strictPTotal
		articleSourceStrictR, articleContentStrictR, articleCueStrictR = strictRTotal

		articleSourceSoftP, articleContentSoftP, articleCueSoftP = softPTotal
		articleSourceSoftR, articleContentSoftR, articleCueSoftR = softRTotal

		'''
		countPredicted += len(finalArticle.attributions)
		print softPTotal[0]/countPredicted
		countGold += len(goldArticle.attributions)
		print softRTotal[0]/countGold


		
		j = j + 1

		if j == maxNumFiles:
			break

	#print countPredicted
	#print countGold

	#SOURCE CONTENT CUE

	print countPredicted
	print countGold

	print allAttrs

	print 'discontinous: ' + str(countDiscontinous)
	print 'repeats: ' + str(countRepeat)


	print "-----------PARTIALS----------"
	print 'Content Scores'
	print str(partialsPTotal[1]/countPredicted)
	print str(partialsRTotal[1]/countGold)
	print str(calculateF(partialsPTotal[1]/countPredicted, partialsRTotal[1]/countGold))

	print 'Source Scores'
	print str(partialsPTotal[0]/countPredicted)
	print str(partialsRTotal[0]/countGold)
	print str(calculateF(partialsPTotal[0]/countPredicted, partialsRTotal[0]/countGold))


	print 'Cue Scores'
	print str(partialsPTotal[2]/countPredicted)
	print str(partialsRTotal[2]/countGold)
	print str(calculateF(partialsPTotal[2]/countPredicted, partialsRTotal[2]/countGold))

	print
	print "-----------STRICTS----------"
	print 'Content Scores'
	print str(strictPTotal[1]/countPredicted)
	print str(strictRTotal[1]/countGold)
	print str(calculateF(strictPTotal[1]/countPredicted, strictRTotal[1]/countGold))

	print 'Source Scores'
	print str(strictPTotal[0]/countPredicted)
	print str(strictRTotal[0]/countGold)
	print str(calculateF(strictPTotal[0]/countPredicted, strictRTotal[0]/countGold))


	print 'Cue Scores'
	print str(strictPTotal[2]/countPredicted)
	print str(strictRTotal[2]/countGold)
	print str(calculateF(strictPTotal[2]/countPredicted, strictRTotal[2]/countGold))

	print
	print "-----------SOFTS----------"
	print 'Content Scores'
	print str(softPTotal[1]/countPredicted)
	print str(softRTotal[1]/countGold)
	print str(calculateF(softPTotal[1]/countPredicted, softRTotal[1]/countGold))

	print 'Source Scores'
	print str(softPTotal[0]/countPredicted)
	print str(softRTotal[0]/countGold)
	print str(calculateF(softPTotal[0]/countPredicted, softRTotal[0]/countGold))


	print 'Cue Scores'
	print str(softPTotal[2]/countPredicted)
	print str(softRTotal[2]/countGold)
	print str(calculateF(softPTotal[2]/countPredicted, softRTotal[2]/countGold))





###print the very final score for the whole dataset


def writeToXml(typeFile, filename, finalArticle):
	xml_string = finalArticle.get_parc_xml(indent='  ')	
	open(os.path.join(xml_dir, typeFile + '/labelled_' + filename), 'w').write(xml_string)

	attributions = finalArticle.attributions
	htmlString = ''
	
	for attr in attributions:
		attribution = finalArticle.attributions[attr]
		html = finalArticle.get_attribution_html(attribution)
		print html
		htmlString += html + '<br>'
	htmlPage = finalArticle.wrap_as_html_page(htmlString)

	open(os.path.join(xml_dir, typeFile + '/html/labelled_' + filename + '.html'), 'w').write(htmlPage)

def overlap(articleComparing, articleCompare, partials, stricts, softs, gold):

	totalSourceProportions, totalContentProportions, totalCueProportions = partials
	totalSourceStrict, totalContentStrict, totalCueStrict = stricts
	totalSourceSoft, totalContentSoft, totalCueSoft = softs

	articleDenom = articleComparing.attributions
	articleCompared = articleCompare.attributions

	for attrID in articleDenom:
		currentAttr = articleDenom[attrID]
		topAttr = 0.0
		topAttrId = None
		proportionContent = None
		overlapContentSoft = None
		overlapContentStrict = None

		if gold == True:
			for attrIDComp in articleCompared:

				compareAttr = articleCompared[attrIDComp]
				proportionContentTemp, overlapContentSoftTemp, overlapContentStrictTemp = compareContents(currentAttr, compareAttr)
				if proportionContentTemp > topAttr:
					topAttr = proportionContentTemp
					topAttrId = compareAttr
					proportionContent, overlapContentSoft, overlapContentStrict = compareContents(currentAttr, compareAttr)
			if topAttrId is not None:
				partials, soft, strict = compareSourceCue(currentAttr, topAttrId)

				proportionSource, proportionCue = partials
				overlapSourceSoft, overlapCueSoft = soft
				overlapSourceStrict, overlapCueStrict = strict


				totalSourceProportions += proportionSource
				totalContentProportions += proportionContent
				totalCueProportions += proportionCue


				totalSourceStrict += overlapSourceStrict
				totalContentStrict += overlapContentStrict
				totalCueStrict += overlapCueStrict

				totalSourceSoft += overlapSourceSoft
				totalContentSoft += overlapContentSoft
				totalCueSoft += overlapCueSoft

				print 'current'
				print currentAttr
				print 'compare'
				print topAttrId
				print currentAttr['content'][0]['sentence_id']


			else:
				print 'none found'
				print currentAttr
				print currentAttr['content'][0]['sentence_id']

				continue
				
		else:
			for attrIDComp in articleCompared:
				compareAttr = articleCompared[attrIDComp]
				proportionContent, overlapContentSoft, overlapContentStrict = compareContents(currentAttr, compareAttr)
				if overlapContentSoft == 1:
					partials, soft, strict = compareSourceCue(currentAttr, compareAttr)

					proportionSource, proportionCue = partials
					overlapSourceSoft, overlapCueSoft = soft
					overlapSourceStrict, overlapCueStrict = strict


					totalSourceProportions += proportionSource
					totalContentProportions += proportionContent
					totalCueProportions += proportionCue

					totalSourceStrict += overlapSourceStrict
					totalContentStrict += overlapContentStrict
					totalCueStrict += overlapCueStrict

					totalSourceSoft += overlapSourceSoft
					totalContentSoft += overlapContentSoft
					totalCueSoft += overlapCueSoft

					print currentAttr
					print compareAttr

					
				else:
					continue

	return (totalSourceProportions, totalContentProportions, totalCueProportions), \
					(totalSourceStrict, totalContentStrict, totalCueStrict), (totalSourceSoft, totalContentSoft, totalCueSoft)


def compareSourceCue(firstAttr, secondAttr):
	overlapSource = 0
	overlapCue = 0

	overlapSourceSoft = 0
	overlapCueSoft = 0

	overlapSourceStrict = 1
	overlapCueStrict = 1

	sourceFirst = firstAttr['source']
	sourceSecond = secondAttr['source']
	for token in sourceFirst:
		foundToken = False
		for token2 in sourceSecond:
			if token['id'] == token2['id'] and token['sentence_id'] == token2['sentence_id']:
				overlapSource += 1
				overlapSourceSoft = 1
				foundToken = True
				break
		if foundToken == False:
			overlapSourceStrict = 0



	cueFirst = firstAttr['cue']
	cueSecond = secondAttr['cue']
	for token in cueFirst:
		foundToken = False
		for token2 in cueSecond:
			if token['id'] == token2['id'] and token['sentence_id'] == token2['sentence_id']:
				overlapCue += 1
				overlapCueSoft = 1
				foundToken = True
				break
		if foundToken == False:
			overlapCueStrict = 0


	#in case anything is empty
	if len(sourceFirst) == 0 and len(sourceSecond) == 0:
		proportionSource = 1
		overlapSourceSoft = 1
		overlapSourceStrict = 1
	elif len(sourceFirst) == 0:
		proportionSource = 0
		overlapSourceSoft = 0
		overlapSourceStrict = 0
	else:
		proportionSource =  float(overlapSource) / len(sourceFirst)


	if len(cueFirst) == 0 and len(cueSecond) == 0:
		proportionCue = 1
		overlapCueSoft = 1
		overlapCueStrict = 1

	elif len(cueFirst) == 0:
		proportionCue = 0
		overlapCueSoft = 0
		overlapCueStrict = 0
	else:
		proportionCue = float(overlapCue) / len(cueFirst)

	
	#if proportionSource < 0.85:
	#	print firstAttr
	#	print secondAttr


	return (proportionSource, proportionCue), (float(overlapSourceSoft), float(overlapCueSoft)), (float(overlapSourceStrict), float(overlapCueStrict))


def compareContents(firstAttr, secondAttr):
	overlapContent = 0
	
	overlapContentSoft = 0

	overlapContentStrict = 1
	

	contentFirst = firstAttr['content']
	contentSecond = secondAttr['content']
	for token in contentFirst:
		foundToken = False
		for token2 in contentSecond:
			if token['id'] == token2['id'] and token['sentence_id'] == token2['sentence_id']:
				overlapContent += 1
				overlapContentSoft = 1
				foundToken = True
				break
		if foundToken == False:
			overlapContentStrict = 0


	
	if len(contentFirst) == 0 and len(contentSecond) == 0:
		proportionContent = 1
		overlapContentSoft = 1
		overlapContentStrict = 1
	elif len(contentFirst) == 0 or len(contentSecond) == 0:
		proportionContent = 0
		overlapContentSoft = 0
		overlapContentStrict = 0
	else:
		proportionContent =  float(overlapContent) / len(contentFirst)

	return proportionContent, float(overlapContentSoft), float(overlapContentStrict)




def calculateF(p, r):
	return (2*p*r)/(p+r)

	




			
#labels rows
def operateCRFSuite(rows, outputFile):

	print 'labelling data'
	#open(os.path.join(data_dir, outputFile), 'w').close()

	

	#go through all rows, convert it to a crf compatible format, tag it, append it to csv file

	lastLabel = 'O'
	labels = []
	#with open(os.path.join(data_dir, outputFile), "a") as myfile:

	fileFeatures = []
	metadataList = []
	for line in rows:
		row = line.split('\t')
		del row[0]
		metadata = ''
		features = {}		
		
		for elem in row:
			seperated = elem.split('=')
			nameFeat = seperated[0]
			if nameFeat != '':
				answer = seperated[1]

			if nameFeat in ['filename', 'sentenceID', 'tokenID']:
				metadata = metadata + elem + ';'

			features.update( {
				nameFeat : answer
			})

		fileFeatures.append(features)
		metadataList.append(metadata)

	allLabels = tagger.tag(fileFeatures)

	returnlabels = []
	for indx, label in enumerate(allLabels):
		metadata = metadataList[indx]
		returnlabels.append(label + '\t' + metadata)


			#myfile.write(label[0] + '\t' + metadata + '\n')
	#myfile.close()
	return returnlabels

def openFile(coreNLPFileName, annotatedFileName, raw_file):


	#open annotated if it exists
	if annotatedFileName != None:
		try:
			parc_xml = open(annotatedFileName).read()
			corenlp_xml = open(coreNLPFileName).read()
			raw_text = open(raw_file).read()
			article = P(corenlp_xml, parc_xml, raw_text)


		except:
			print 'error opening file'
			raise
			return -1

	else:
		parc_xml = None
		corenlp_xml = open(coreNLPFileName).read()
		raw_text = open(raw_file).read()
		article = P(corenlp_xml, parc_xml, raw_text)

	filename = coreNLPFileName.split('/')[-1]

	return article

def remove_attribution1(article, attribution_id):
		'''
		Deletes the attribution identified by attribution_id, including
		all references from sentences, tokens, and globally
		'''
		attribution = article.attributions[attribution_id]

		# first remove the attribution from each of the tokens
		sentence_ids = set()
		tokens = (
			attribution['cue'] + attribution['content'] 
			+ attribution['source']
		)
		for token in tokens:
			sentence_ids.add(token['sentence_id'])
			token['role'] = None
			token['attribution'] = None

		# Delete references to the attribution on sentences
		for sentence_id in sentence_ids:
			sentence = article.sentences[sentence_id]
			if attribution_id in sentence['attributions']:
				del sentence['attributions'][attribution_id]

		# Delete the global reference to the attribution
		del article.attributions[attribution_id]

#parse command line arguments
def main():
	usageMessage = '\nCorrect usage of the Labeling command is as follows: \n' + \
					'python source/testData.py /pathToCoreNLPDirectory /pathToAnnotated /pathToRaw outputFile \n' + \
					'\nFor reference, the path to the CoreNLP file is: /home/ndg/dataset/ptb2-corenlp/CoreNLP/ + train, test or dev depending on your needs. \n' + \
					'\n /home/ndg/dataset/ptb2-corenlp/masked_raw/test/' + \
					'The path to the Parc3 files is /home/ndg/dataset/parc3/ + train, test or dev'

	args = sys.argv

	if len(args) == 5:

		pathToCORENLP = args[1]
		pathToAnnotated = args[2]
		pathToRaw = args[3]
		outputFile = args[4]

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

		if os.path.isdir(pathToAnnotated):
			print 'valid path to a directory'
		else:
			print 'ERROR: The path to this annotated_text directory does not exist.'
			print usageMessage
			return

		label(pathToCORENLP, pathToRaw, pathToAnnotated, outputFile)


	else:
		print usageMessage



if __name__ == '__main__':
   main()


