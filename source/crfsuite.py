import sys
import os
import csv
from statsd import StatsClient
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from itertools import chain

import pycrfsuite


statsd = StatsClient()


print sys.getdefaultencoding()
reload(sys)
sys.setdefaultencoding('utf-8')

#change this if you would only like to do a certain number of files, useful for testing
maxNumFiles = 1000

#base dir for all data files
data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/'))


def deleteLabel(dictionary):
	del dictionary['label']
	return dictionary

#divide dataset into features and labels
@statsd.timer('formatDataSet')
def formatDataSet(features):


	#Y = [[s['label']] for s in features]
	#X = [[deleteLabel(s)] for s in features]


	Y = [[word['label'] for word in article]for article in features]
	X = [[deleteLabel(word) for word in article]for article in features]


	print len(X)

	return X, Y

#turn features into crfsuite readable object
def word2features(token):

	features = {
		'label' : token[0]
	}

	del token[0]

	for elem in token:
		seperated = elem.split('=')
		nameFeat = seperated[0]

		#if nameFeat == 'minDistanceVerbCue':
		#	continue

		answer = seperated[1]

		features.update( {
			nameFeat : answer
		})

	return features

#creates a report for BIO encoded sequences
def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_)
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )


#trains a classifier based on content token features
def trainAll(X_train, y_train):
	crf = pycrfsuite.Trainer(
		verbose = True,
		#algorithm = 'l2sgd',
	)

	crf.set_params({
    'max_iterations': 40,  # stop earlier
    'feature.minfreq': 2,
    'feature.possible_transitions': False
	})

	for xseq, yseq in zip(X_train, y_train):
		crf.append(xseq, yseq)

	crf.train('ContentSpanClassifier8.crfsuite')
	return crf


def trainData():
	lines = []
	trainingFileName = os.path.join(data_dir, 'PARCTrainContentSpans2.txt')
	reader_object = open(trainingFileName, 'r')
	lines = reader_object.readlines()

	print 'length of training set'
	print len(lines)

	allFeatures = []
	thisFileFeatures = []

	print 'extracting features'

	lastfilename = None

	i = 0
	for line in lines:
		i = i + 1
		row = line.split('\t')
		features = word2features(row)
		filename = features['filename']
		if filename == lastfilename or lastfilename == None:
			thisFileFeatures.append(features)
			lastfilename = filename
		else:
			allFeatures.append(thisFileFeatures)
			thisFileFeatures = []
			thisFileFeatures.append(features)
			lastfilename = filename

	print len(allFeatures)


	print 'features extracted'

	print 'formatting data set'

	x_train, y_train = formatDataSet(allFeatures)
	prevPred = ['O']
	for pred in y_train:
		if pred == ['I'] and prevPred == ['O']:
			print 'foundTRAIN'

		prevPred = pred


	print 'trainingData'
	#classifier = TRAIN(x_train, y_train, x_test, y_test)
	classifier = trainAll(x_train, y_train)

#tests the results of a classifier against a labelled dataset
def test(X_test, y_test):
	tagger = pycrfsuite.Tagger()
	#tagger.open('ContentSpanClassifier.crfsuite')
	tagger.open('ContentSpanClassifier8.crfsuite')

	print 'new'
	y_pred2 = [tagger.tag(xseq) for xseq in X_test]
	prevPred = 'O'
	for pred in y_pred2:
		if pred == 'I' and prevPred == 'O':
			print 'foundTEST'

		prevPred = pred

	print(bio_classification_report(y_test, y_pred2))
	y_test2 = [item for sublist in y_test for item in sublist]

	y_pred3 = [item for sublist in y_pred2 for item in sublist]

	print accuracy_score(y_test2, y_pred3)

#tests the classifier that is created against some data
def testData():
	testingFileName = data_dir + '/PARCTestContentSpans1.txt'
	reader_object = open(testingFileName, 'r')
	lines = reader_object.readlines()

	print 'length of test set'
	print len(lines)

	allFeatures = []
	thisFileFeatures = []

	print 'extracting features'

	lastfilename = None

	i = 0
	for line in lines:
		i = i + 1
		row = line.split('\t')
		features = word2features(row)
		filename = features['filename']
		if filename == lastfilename or lastfilename == None:
			thisFileFeatures.append(features)
			lastfilename = filename
		else:
			allFeatures.append(thisFileFeatures)
			thisFileFeatures = []
			thisFileFeatures.append(features)
			lastfilename = filename

	print len(allFeatures)


	print 'features extracted'

	print 'formatting data set'

	x_test, y_test= formatDataSet(allFeatures)
	test(x_test, y_test)


def main():
	print sys.argv
	if sys.argv[1] == '-test':
		testData()
	elif sys.argv[1] == '-train':
		trainData()
	else:
		print 'Use of this command line is: python source/crfsuiteTests.py -test or -train'
	#labelData()	

if __name__ == '__main__':
   main()