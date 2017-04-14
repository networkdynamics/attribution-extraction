from nltk.corpus import wordnet as wn 
import csv


def getHypoIter(string):
	listOfPeople = []

	synsets = wn.synsets(string)

	head = []
	tail = []

	for synset in synsets:

		listOfPeople = listOfPeople + [synset]
		hypo = synset.hyponyms()

		if len(hypo) != 0:
			head = hypo[0]
			tail = hypo[1:]

		
			while (len(tail) != 0):


				newHypo = head.hyponyms()
				listOfPeople = listOfPeople + [head] 


				tail = tail + newHypo
				head = tail[0]
				tail = tail[1:]


		else:
			continue
	print len(listOfPeople)
	print listOfPeople
	return listOfPeople

def getHypoIterNominals(string):
	finalList = []

	synset = wn.synset(string)

	head = []
	tail = []

	finalList = finalList + [synset]
	hypo = synset.hyponyms()

	if len(hypo) != 0:
		head = hypo[0]
		tail = hypo[1:]
	
		while (len(tail) != 0):
			newHypo = head.hyponyms()
			finalList = finalList + [head] 


			tail = tail + newHypo
			head = tail[0]
			tail = tail[1:]
	else:
		print len(finalList)
		return finalList

	return finalList

def getNominals():
	wordList = ['person.n.01', 'group.n.01', 'communication.n.02']

	fullList = []

	for word in wordList:
		thisList = getHypoIterNominals(word)
		fullList += thisList

	formattedList = format(fullList)

	print len(formattedList)

	formattedList = list(set(formattedList))
	formattedList.sort()

	print len(formattedList)

	with open('allHypos.csv', 'w') as allHypos:
		writer = csv.writer(allHypos, delimiter = ',')
		writer.writerow(formattedList)


def format(listOfSyn):

	finalList = []

	for synset in listOfSyn:
		word = synset.name().split(".")[0].replace('_',' ')
		finalList = finalList + [word]

	return finalList

def getLists():
	listOfPeople = format(getHypoIter('person'))
	listofOrgs = format(getHypoIter('organization'))

	with open('peopleHyponyms.csv', 'w') as people:
		writer = csv.writer(people, delimiter = ',')
		writer.writerow(listOfPeople)

	with open('orgHyponyms.csv', 'w') as organization:
		writer = csv.writer(organization, delimiter = ',')
		writer.writerow(listofOrgs)

	return listOfPeople, listofOrgs




def main():
	#getLists()
	getNominals()



if __name__ == '__main__':
   main()
