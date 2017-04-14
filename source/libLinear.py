import numpy as np
import pandas as pd 
import os
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from time import time
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import operator
from sklearn.externals import joblib
import sys

data_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/'))


def testData():
	logreg = joblib.load('sourceTrainer2.pkl') 

	#df = pd.read_csv(os.path.join(data_dir, "sourceEntityTrainingSet1.csv"), header=None, usecols=range(34))
	#df = pd.read_csv(os.path.join(data_dir, "PARCdevSourceSpanFeatures.csv"), header=None, usecols=range(34))
	df = pd.read_csv(os.path.join(data_dir, "testSpanFeatures7.csv"), header=None, usecols=range(35))
	#df = pd.read_csv(os.path.join(data_dir, "sourceEntityMatchDev1.csv"), header=None, usecols=range(38))
	#df = pd.read_csv(os.path.join(data_dir, "xab"), header=None, usecols=range(38))

	labels = 'containsPerson,featNextPunctE,featNextPunctQ,featNextQuoteE,featNextSpeakerE,featNextSpeakerQ,featNextVerbE,featNextVerbQ,featPrevPunctE,featPrevPunctQ,featPrevQuoteE,featPrevSpeakerE,featPrevSpeakerQ,featPrevVerbE,featPrevVerbQ,isNSubj,isNSubjVC,numEntityBetween,numMentionsPrevPar,numMentionsThisPar,numQuotesAttributed,numQuotesBetween,numQuotesOtherSpeakers,numQuotesParag,numQuotesPrev9Parag,numWordsBetween,numWordsParag,numWordsPrev9Parag,otherSpeakerMentioned,quoteDistanceFromParag,sameSentence,thisSpeakerMentioned,wordLengthQuote'

	df[0] = pd.Categorical(df[0]).codes


	df[1] = df[1].map({'Y': 1, 'N': 0})
	print labels
	print df


	df = df.sort_values(by = 0, ascending=False)
	df = df.values

	newFeats = np.split(df, np.where(np.diff(df[:,0]))[0]+1)
	y_test = [[feat[:, 1]] for feat in newFeats]

	x_test = newFeats
	print logreg.coef_

	maxcoef = np.argmax(logreg.coef_[0])
	print maxcoef



	labels = labels.split(',')

	print "most important feat"
	print labels[maxcoef]

	print "all feats sorted"
	indicesSorted = np.argsort(logreg.coef_[0])
	print indicesSorted
	for index in indicesSorted:
		print labels[index]



	flatXtest = np.array([item for sublist in x_test for item in sublist])
	print flatXtest[:3]
	flatYtest = flatXtest[:,1]

	flatXtest = np.delete(flatXtest, [0,1], 1)


	t0 = time()
	print("Predicting entity")
	y_pred = logreg.predict(flatXtest)
	print y_pred[:10]
	print("done in %0.3fs" % (time() - t0))

	print(classification_report(flatYtest, y_pred))
	print(confusion_matrix(flatYtest, y_pred))


	print x_test[0]
	print y_test[0]

	print len(x_test[0])
	print len(y_test[0])

	total = 0
	correct = 0

	for i, elem in enumerate(x_test):
		elem = np.delete(elem, [0,1], 1)
		arrayProbas = logreg.predict_proba(elem)
		# ... compute some result based on item ...
		positics = np.delete(arrayProbas, 0, 1)

		maxval = np.argmax(positics)

		pred = y_test[i][0][maxval]

		if pred == 1:
			correct = correct + 1
		
		total = total + 1

	print correct
	print total

	print float(correct)/float(total)



#find a way to print actual prediction score
#for how many quotes is the top prediction score correct 
def trainData():
	#df = pd.read_csv(os.path.join(data_dir, "sourceEntityTrainingSet1.csv"), header=None, usecols=range(34))
	#df = pd.read_csv(os.path.join(data_dir, "PARCtrainSourceSpanFeatures.csv"), header=None, usecols=range(34))
	df = pd.read_csv(os.path.join(data_dir, "trainSpanFeatures7.csv"), header=None, usecols=range(35))
	#df = pd.read_csv(os.path.join(data_dir, "xaa"), header=None, usecols=range(38))


	df[0] = pd.Categorical(df[0]).codes
	print 'here'


	df[1] = df[1].map({'Y': 1, 'N': 0})
	print df


	df = df.sort_values(by = 0, ascending=False)

	df = df.values

	newFeats = np.split(df, np.where(np.diff(df[:,0]))[0]+1)
	y_train = [[feat[:, 1]] for feat in newFeats]


	x_train = newFeats

	flatXtrain = np.array([item for sublist in x_train for item in sublist])
	flatYtrain = flatXtrain[:,1]

	flatXtrain = np.delete(flatXtrain, [0,1], 1)


	logreg = linear_model.LogisticRegression()
	logreg = logreg.fit(flatXtrain, flatYtrain)

	joblib.dump(logreg, 'sourceTrainer2.pkl')


def main():
	print sys.argv
	if sys.argv[1] == '-test':
		testData()
	elif sys.argv[1] == '-train':
		trainData()
	else:
		print 'Use of this command line is: python source/libLinearTests.py -test or -train'
	#labelData()	

if __name__ == '__main__':
   main()



