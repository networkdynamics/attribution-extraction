import pdb

def resolveSourceSpan(article, fileRows, perms):

	#return if no attributions in this article
	if perms == []:
		return article

	articleSentences = article.sentences
	finalHead = ''
	print 'source span resolving'

	#for eacha attribution, get the entity, quote, and relevent sentences
	for perm in perms:
		if getattr(perm, 'label') == 'N':
			continue
		else:

			entity = getattr(perm, 'entityObject')
			print entity	
			quote = getattr(perm, 'quoteObject')
			print quote
			sentID = entity['sentence_id']
			attrID = quote['attribution_id']
			#article.attributions[attrID]['source'] = entity['tokenArray']
			#continue

			currSent = articleSentences[sentID]


			tokens = entity['tokenArray']

			attr = quote['attribution_id']
			#if attr == 'my_attribution_wsj_2393_Attribution_relation_level.xml_set_5_wsj_2393.xml0':
			#	pdb.set_trace()
			#article.attributions[attr]['source'] = tokens
			#continue

			#find head of entity tokenArray
			entityHead = article._find_head(tokens)
			#print entityHead
			#print tokens

			finalHead = ''
			#find the noun
			for token in entityHead:
				posHead = token['pos']
				if posHead.startswith('N') or posHead == 'PRP' or posHead.startswith('W'):
					finalHead = token
					break
			#if there is no noun, we're fucked
			if finalHead == '':
				for token in entityHead:
					newEntityHead = entityHead[0]['parents']
					if newEntityHead != []:
						finalHead = newEntityHead[0][1]

			#if this whole process didn't work, let's figure out why
			if finalHead == '':
				finalHead = tokens[0]
				#attr = quote['attribution_id']
				#article.remove_attribution(attr)
				#continue

			sentID = entity['sentence_id']
			sentIDQuote = quote['sentence_id']

			currSent = articleSentences[sentID]

			sourceSpanTokens = [finalHead]

			s = Stack()

			for parent in finalHead['parents']:
				if 'compound' in parent[0] or 'nmod' in parent[0]:
					sourceSpanTokens.append(parent[1])
					s.push(parent[1])



			if finalHead.has_key('children'):
				s.push(finalHead)
				while not s.isEmpty():
					currElem = s.pop()
					#print currElem.keys()
					if currElem.has_key('children'):
						theseChildren = currElem['children']
						#print theseChildren
						for child in theseChildren:
							sourceSpanTokens.append(child[1])
							s.push(child[1])

			

			#print sourceSpanTokens

			sourceSpanArray = []

			for token in currSent['tokens']:
				quoteTokens = quote['tokenArray']

				inQuote = indexOf(token, quoteTokens)
				inSource = indexOf(token, sourceSpanTokens)

				if inQuote:
					continue
				elif inSource:
					sourceSpanArray.append(token)
					continue

			attrID = quote['attribution_id']
			quoteArray = quote['tokenArray']
			if sourceSpanArray == []:
				article.attributions[attrID]['source'] = []
				continue

			sourceSpanBegin = sourceSpanArray[0]['id']
			sourceSpanEnd = sourceSpanArray[-1]['id']

			sentenceID = sourceSpanArray[0]['sentence_id']
			for token in tokens:
				if 'role' in token and token['role'] == 'content':
					print 'ALERT'
			tokens = article.sentences[sentenceID]['tokens'][sourceSpanBegin:sourceSpanEnd + 1]

		
			#for now removing and readding, until figure out how to update
			try:
				article.attributions[attrID]['source'] = tokens
			except Exception, e:
				print e
	
	return article

def indexOf(token, target):
	for aToken in target:
		if aToken['id'] == token['id'] and aToken['sentence_id'] == token['sentence_id']:
			return True

	return False


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



