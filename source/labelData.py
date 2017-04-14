#this program runs the pipeline and then compares the predicted results with gold spans 

print 'importing'
import contentSpanExtractor as contentSpans
import intermediaries.kNN as kNN
import sourceFeatureExtraction as resolveSource
import sourceEntityTrainingSet as tokenize
import resolveSourceSpan as sourceSpanning
import resolveCue as cueResolving
#import crfsuite as crf
from parc_reader import ParcCorenlpReader as P
import sys
import os
import csv
import pycrfsuite
import verbCuesFeatureExtractor as verbCues
import cProfile
from memory_profiler import profile
import gc
from SETTINGS import PACKAGE_DIR

data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../', 'data/'))
xml_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../', 'data/xmls/'))

maxNumFiles = -1
#CuetrainingFile = os.path.join(data_dir, 'VerbFeatsForKNN.tsv')
CuetrainingFile = os.path.join(data_dir, 'PARCTrainVerbFeats.csv')
CuetrainingData = kNN.readData(CuetrainingFile, False)

tagger = pycrfsuite.Tagger()
tagger.open(os.path.join(PACKAGE_DIR, 'ContentSpanClassifier3.crfsuite'))

with open(data_dir + '/nounCues.csv', 'rb') as f:
    reader = csv.reader(f)
    nounCues = list(reader)


#open all files, run pipeline, check metrics
def label(path):
	#open files
	listOfNLPFiles = verbCues.openDirectory(path)
	length = str(len(listOfNLPFiles))
	#listOfRawFiles = verbCues.openDirectory(pathToRaw)
	#listOfDirFiles = verbCues.openDirectory(outputDir)

	#[strict, partialPrecision, partialRecall, partialFscore, soft]
	j = 0
	for myFile in listOfNLPFiles:
		if j < 2250:
			j += 1
			continue
		sourceDir = '/home/ndg/dataset/'
		sourceDir = '/home/ndg/dataset/political-validators/nyt'
		print myFile
		labelInd(myFile, sourceDir)
	###compare perfect article with predicted article and create a score
		
		#ps and rs scores
		j = j + 1
		print str(j) + 'of' + length
		if j == maxNumFiles:
			break
		gc.collect()

def labelInd(myFile, sourceDir):

	filename = os.path.basename(myFile)
	#typeFile = myFile.split('/')[-2]
	fileNoXML = filename.rsplit('.xml',1)[0]

	filename = fileNoXML
	print filename		

	#print('opening file: ' + myFile + ' ' + str(j) + ' out of ' + str(files))

	corenlpPath = sourceDir + '/corenlp/' + filename + '.xml'
	rawPath = sourceDir + '/article-text-filtered/' + filename 
	outputDir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../', 'nyt/'))

	#corenlpPath = sourceDir + 'ptb2-corenlp/CoreNLP/test/' + filename + '.xml'
	#rawPath = sourceDir + 'ptb2-corenlp/masked_raw/test/' + filename
	#outputDir = data_dir + '/attributions/'

###resolve to raw and annotated files
	#myRawFile = [s for s in listOfRawFiles if fileNoXML in s][0]

###find verb cues and label with KNN


		#labelledVerbs = newkNN.labelWithoutCSV(CuetrainingData, thisFilesVerbs)
		#print labelledVerbs
		#REVISIT
		
	###comment or uncomment to use predicted spans or gold spans and then tokenize the file into <quote> <entity> etc.

		#PREDICTED SPANS
		#'''
	try:	
		thisFilesVerbs, article = verbCues.openFile(corenlpPath, None, rawPath)
		print 'FINDING VERBS'	
		labelledVerbs = kNN.labelWithoutCSV(CuetrainingData, thisFilesVerbs)

		print 'FINDING CONTENT SPANS'
		fileRows = contentSpans.findFeatures(filename, article, labelledVerbs, None)
		contentSpanLabels = operateCRFSuite(fileRows)
		print 'FINDING SOURCE ENTITIES'
		tokenizedRows, newArticle = tokenize.process(article, filename, contentSpanLabels, labelledVerbs)

		
	###decide which entities are the best entity matches
		#REVISIT
		print 'RESOLVING CUES AND '
		theseTokens = resolveSource.postprocess(tokenizedRows, article, False)

	###find the whole source span
		#REVISIT
		newArticle = sourceSpanning.resolveSourceSpan(newArticle, tokenizedRows, theseTokens)
	###find the whole cue span
		#REVISIT
		finalArticle = cueResolving.linkCue(newArticle, labelledVerbs, nounCues)
		#for attr in finalArticle.attributions:
		#	print finalArticle.attributions[attr]

		
		#for attr in finalArticle.attributions:
		#	print finalArticle.attributions[attr] 
			
		writeToXml(outputDir, filename, finalArticle)
	except Exception, e:
		print e
	

###print the very final score for the whole dataset

###print the very final score for the whole dataset

def writeToXml(outputDir, filename, finalArticle):
	xml_string = finalArticle.get_parc_xml(indent='  ')	
	out_file = open(os.path.join(outputDir, 'labelled_' + filename), 'w')
	out_file.write(xml_string)
	out_file.close()

	#open(os.path.join(outputDir, 'labelled_' + filename), 'w').write(xml_string).close()

	html = finalArticle.get_all_attribution_html()
	
	#attributions = finalArticle.attributions
	#htmlString = ''
	
	#for attr in attributions:
	#	attribution = finalArticle.attributions[attr]
	#	html = finalArticle.get_attribution_html(attribution)
	#	htmlString += html + '<br>'
	
	#htmlPage = finalArticle.wrap_as_html_page(html)

	#open(os.path.join(outputDir, 'html/labelled_' + filename + '.html'), 'w').write(html)

			
#labels rows
def operateCRFSuite(rows):

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
		row = row[1:]
		metadata = ''
		features = {}		
		
		for elem in row:
			seperated = elem.split('=')
			nameFeat = seperated[0]
			if nameFeat != '':
				answer = seperated[1]

			if nameFeat in ['filename', 'sentenceID', 'tokenID']:
				metadata = metadata + elem + ';'

			if nameFeat == 'minDistanceVerbCue':
				continue

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



#parse command line arguments
def main():
	usageMessage = '\nCorrect usage of the Labeling command is as follows: \n' + \
					'python source/labelData.py /pathToCoreNLPDirectory /pathToAnnotated /pathToRaw outputDirectory \n' + \
					'\nFor reference, the path to the CoreNLP file is: /home/ndg/dataset/ptb2-corenlp/CoreNLP/ + train, test or dev depending on your needs. \n' + \
					'\n /home/ndg/dataset/ptb2-corenlp/masked_raw/test/' + \
					'The path to the Parc3 files is /home/ndg/dataset/parc3/ + train, test or dev'

	args = sys.argv
	print args

	if len(args) == 4:

		pathToCORENLP = args[1]
		pathToRaw = args[2]
		outputDir = args[3]

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


		if os.path.isdir(outputDir):
			print 'valid path to a directory'
		else:
			print 'ERROR: The path to this output directory does not exist.'
			print usageMessage
			return

		#label(pathToCORENLP, pathToRaw, outputDir)
		#pr = cProfile.Profile()
		#pr.enable()
		label(pathToCORENLP)
		#labelReverse(pathToCORENLP, pathToRaw, outputDir)
		#pr.disable()
		#pr.print_stats(sort='time')
	elif len(args) == 2:
		pathToCORENLP = args[1]
		label(pathToCORENLP)
	else:
		print usageMessage




if __name__ == '__main__':
   main()


