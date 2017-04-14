from bs4 import BeautifulSoup as Soup


class AnnotatedText(object):

	def __init__(self, parc_xml):

		self.soup = Soup(parc_xml, 'html.parser')

		self.words = []
		self.sentences = []
		sentence_tags = self.soup.find_all('sentence')
		for sentence_tag in sentence_tags:
			sentence = {'words':[]}
			self.sentences.append(sentence)
			word_tags = sentence_tag.find_all('word')
			for word_tag in word_tags:

				thisword = word_tag['text']

				if thisword == '-LRB-':
					thisword = '('
				if thisword == '-RRB-':
					thisword = ')'
				if "\\/" in thisword:
					thisword = '/'.join(thisword.split("\\/"))

				word = {
					'token': thisword.encode('utf-8'),
				}
				attribution = word_tag.find('attribution')
				if attribution:
					word['attribution'] = {
						'role': attribution.find('attributionrole')['rolevalue'],
						'id': attribution['id']
					}
				else:
					word['attribution'] = None

				self.words.append(word)
				sentence['words'].append(word)
