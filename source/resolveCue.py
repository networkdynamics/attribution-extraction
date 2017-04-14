from operator import itemgetter

def linkCue(article, verbs, nouns):
	attrs = article.attributions
	sentences = article.sentences

	keys = attrs.keys()


	print 'cueresolving'

	for key in keys:
		currentAttr = attrs[key]
		contentSpan = currentAttr['content']
		sourceSpan = currentAttr['source']

		if contentSpan == []:
			continue
		contentSent = contentSpan[0]['sentence_id']
		thiSentence = sentences[contentSent]
		#print sentences[contentSent]	
		#print contentSpan
		#print sourceSpan

		contentHead = find_head(contentSpan)
		sourceHead = find_head(sourceSpan)

		contentParent = []
		if len(contentHead) > 0:
			for head in contentHead:
				if 'parents' in head:
					contentParent = contentParent + head['parents']		

		sourceParent = []
		sourceSent = -1
		if len(sourceSpan) > 0:
			boolean, source, cue = accordingTos(sourceSpan, currentAttr)
			if boolean == True:
				assign(article, currentAttr, source, cue)
				continue

		sourceParent = []
		for head in sourceHead:
			if 'parents' in head:
				sourceParent = sourceParent + head['parents']	

		assigned = False
		if len(contentParent) >= 1:
			for tok in contentParent:
				thisTok = tok[1]
				
				if isVerbCue(thisTok, verbs):
					tokens = getVerbAux(thisTok)
					assign(article, currentAttr, sourceSpan, tokens)
					assigned = True
					break
					#print currentAttr

		if assigned == False and len(sourceParent) >= 1:
			for tok in sourceParent:
				thisTok = tok[1]
				
				if isVerbCue(thisTok, verbs):
					#print 'content', thisTok
					tokens = getVerbAux(thisTok)
					assign(article, currentAttr, sourceSpan, tokens)
					assigned = True
					break
		
		if assigned == True:
			continue


		if  currentAttr['source'] != [] and hasVerbCue(sourceSent, verbs):
			token = closestVerbCue(currentAttr['source'], verbs, sentences[sourceSent])
			#print 'closest source', token
			tokens = getVerbAux(token)
			assign(article, currentAttr, sourceSpan, tokens)


			#print currentAttr
			continue
		elif hasVerbCue(contentSent, verbs):
			token = closestVerbCue(currentAttr['content'], verbs, thiSentence)
			tokens = getVerbAux(token)
			#print 'verb cue sentence', token

			assign(article, currentAttr, sourceSpan, tokens)


			#print currentAttr
			continue

		ContenthasANounCue = hasNounCue(sentences[contentSent], nouns)
		if sourceSent != -1:
			SourcehasANounCue = hasNounCue(sentences[sourceSent], nouns)

		if sourceSent != -1 and SourcehasANounCue:
			#print 'source sentence noun cue', token

			cue = closestVerbCue(currentAttr['source'], nouns, sentences[sourceSent])
			tokens = getVerbAux(cue)
			assign(article, currentAttr, sourceSpan, tokens)


			continue

		elif ContenthasANounCue:

			cue = closestVerbCue(currentAttr['content'], nouns, sentences[contentSent])
			tokens = getVerbAux(cue)
			#print 'content sentence noun cue', token

			assign(article, currentAttr, sourceSpan, tokens)

			continue



		if tokenSurroundingContent(sentences[contentSent], currentAttr['content'], 'before') != False:
			cue = tokenSurroundingContent(sentences[contentSent], currentAttr['content'], 'before')
			tokens = getVerbAux(cue)
			#print sentences[contentSent], currentAttr['content']
			#print 'next token', tokens

			assign(article, currentAttr, sourceSpan, tokens)

		elif tokenSurroundingContent(sentences[contentSent], currentAttr['content'], 'after') != False:
			cue = tokenSurroundingContent(sentences[contentSent], currentAttr['content'], 'after')
			tokens = getVerbAux(cue)
			#print sentences[contentSent], currentAttr['content']
			#print 'previous token', tokens

			assign(article, currentAttr, sourceSpan, tokens)



		else:
			print 'DIDN"T FIND ANYTHING'

	return article


def assign(article, currentAttr, source, cue):



	sourceSpan = []

	if source == []:
		print source
		print currentAttr
		print cue
		headCue = find_head(cue)
		print headCue
		parents = headCue[0]['children']
		print parents
		for parent in parents:
			if 'nsubj' in parent[0] or 'dobj' in parent[0] and 'role' not in parent[1]:
				source = [parent[1]]




	article.add_to_attribution(currentAttr, 'source', source)
	article.add_to_attribution(currentAttr, 'cue', cue)

	#currentAttr['source'] = source
	#currentAttr['cue'] = cue


def getVerbAux(token):
	if 'children' not in token:
		return [token]

	children = token['children']
	tokenArray = [token]
	for child in children:
		if child[0] == 'auxpass' or child[0] == 'aux' or child[0] == 'neg' or child[0] == 'mark':
			tokenArray.append(child[1])

	tokens = sorted(tokenArray, key=lambda k: k['id'])
	return tokens 





def hasVerbCue(sentenceID, verbs):
	for verb in verbs:
		if int(verb[1]) == sentenceID:
			if verb[4] == 'Y':
				return True
			else:
				continue

	return False

def isVerbCue(verb, verbCues):


	tokenID = verb['id']
	sentID = verb['sentence_id']

	for verb in verbCues:
		if verb[1] == sentID and verb[2] == tokenID and verb[4] == 'Y':
			return True
		elif verb[0].lower() in ['added', 'adds', 'said', 'say', 'says', 'expect', 'expected', 'expects', 'report', 'believe', 'believes', 'reports', 'believed']:
			return True

	return False




def closestVerbCue(content, verbs, sentence):
	sentenceID = sentence['id']

	tokenIds = [token['id'] for token in content]

	cueVerbs = []
	for verb in verbs:
		if int(verb[1]) == sentenceID:
			if verb[4] == 'Y':
				cueVerbs.append(verb)
			else:
				continue
		else:
			continue

	closest = 1000
	closestVerb = ''

	if len(cueVerbs) == 1:
		token = sentence['tokens'][int(cueVerbs[0][2])]
		return token
	else:
		for cueVerb in cueVerbs:
			verbID = int(verb[2])
			distance = min(abs(tokenIds[0] - verbID), abs(tokenIds[-1] - verbID))
			if closest > distance:
				closest = distance
				closestVerb = cueVerb

		return sentence['tokens'][int(closestVerb[2])]


def hasNounCue(thisSentence, nouns):
	for token in thisSentence['tokens']:
		if token['lemma'] in nouns:
			if token.has_key('role') and token['role'] != 'content':
				return True
			else:
				return True

	return False

def closestNounCue(content, nouns, sentence):
	tokenIds = [token['id'] for token in content]

	nounCues = []
	for token in sentence['tokens']:
		if token['lemma'] in nouns:
			if token.has_key('role') and token['role'] == 'content':
				continue
			else:
				nounCues.append(token)

	closest = 1000
	closestNoun = ''

	if len(cueVerbs) == 1:
		token = nounCues[0]
		return token
	else:
		for nounCue in nounCues:
			cueId = nounCue['id']
			distance = min(abs(tokenIds[0] - cueId), abs(tokenIds[-1] - cueId))
			if closest > distance:
				closest = distance
				closestNoun = nounCue

		return closestNoun


def tokenSurroundingContent(sentence, content, position):

	firstTokenID = content[0]['id']
	lastTokenID = content[-1]['id']

	if position == 'before':
		if firstTokenID != 0:
			return sentence['tokens'][firstTokenID - 1]
		else:
			return False
	elif position == 'after':
		if lastTokenID != len(sentence) - 1:
			return sentence['tokens'][firstTokenID + 1]
		else:
			return False
	else:
		return False


def accordingTos(sourceSpan, currentAttr):
	for token in sourceSpan:
		if token['word'].lower() == 'according':
			accordingID = sourceSpan.index(token)
			nextID = accordingID + 1
			if len(sourceSpan) >= nextID + 1 and sourceSpan[accordingID + 1]['word'].lower() == 'to':
				cueSpan = [token, sourceSpan[accordingID + 1]]
				del sourceSpan[nextID]
				del sourceSpan[accordingID]
				return True, sourceSpan, cueSpan

	return False, None, None
	



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
















		


