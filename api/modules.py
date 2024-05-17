import os
import json
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk import ngrams, PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords as stopwords_nltk
stopwords_nltk = stopwords_nltk.words('english')
import pywikibot


# DEFINE


# parameters
nr_mscs_cutoff = 5
nr_keywords_cutoff = 5


# from evaluate_classification import load_index
def load_index():

	# ent_cls_idx
	print('Load entity-class index...')
	try:
		# Docker container
		with open("./data/ent_cls_idx.json", 'r', encoding='utf-8') as f:
			sorted_ent_cls_idx = json.load(f)
	except:
		# Local testing
		with open(os.path.join(os.path.dirname(os.getcwd()), 'data/ent_cls_idx.json'),
				  'r', encoding='utf-8') as f:
			sorted_ent_cls_idx = json.load(f)
	print('...done!')

	return sorted_ent_cls_idx


# get Wikidata qid from name using pywikibot
def get_qid_pywikibot(name):
	try:
		site = pywikibot.Site("en", "wikipedia")
		page = pywikibot.Page(site, name)
		item = pywikibot.ItemPage.fromPage(page)
		qid = item.id
	except:
		qid = None
	return qid

def get_wikidata_qid_wikipedia_url(name):

	verbose = False

	qid = None
	url = None

	try:

		# get wikidata qid
		lang = 'en'
		site = pywikibot.Site(lang, 'wikipedia')
		page = pywikibot.Page(site, name)
		item = pywikibot.ItemPage.fromPage(page)
		qid = item.id

		# get wikipedia url
		try:
			wiki_code = lang + 'wiki'
			page_title = item.sitelinks[wiki_code].title
			# Create the URL manually
			url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
		except Exception as e:
			if verbose:
				print(f'Error: {e}')

	except Exception as e:
		if verbose:
			print(f'Error: {e}')

	return qid, url



def get_keywords(text, get_qids):

	keywords = []
	qids = []

	n_gram_lengths = [2, 3]
	for n in n_gram_lengths:
		try:
			nngrams = ngrams(text.split(), n)
			for nngram in nngrams:
				entity = ''
				for word in nngram:
					entity += word + ' '
				entity = entity[:-1]
				try:
					if sorted_ent_cls_idx[entity] is not None and entity not in stopwords_nltk and entity not in stopwords_custom:
						keywords.append(entity)
				except:
					pass
		except:
			pass

	keywords = list(set(keywords))

	# qid via pywikibot
	if get_qids:
		for keyword in keywords:
			qids.append(get_qid_pywikibot(keyword))

	return keywords, qids

def get_keywords_spans_qids_wikip(text, preprocessing = None):

	# Preprocessing

	if preprocessing == 'stem':
		preprocesser = PorterStemmer()
	elif preprocessing == 'lemma':
		preprocesser = WordNetLemmatizer()

	verbose = False

	keywords = []
	keywords_augmented = []

	# Tokenize the text once (needed for accurate index tracking)
	tokens = text.split()

	n_gram_lengths = [2, 3]
	for n in n_gram_lengths:
		# Generate the n-grams from tokens
		current_ngrams = list(ngrams(tokens, n))

		# Loop through each n-gram to find the valid ones
		for index, nngram in enumerate(current_ngrams):

			# Identify stemming/lemmatization differences
			# if verbose:
			# 	for word in nngram:
			# 		if preprocessing == 'stem':
			# 			preprocessed = preprocesser.stem(word)
			# 		elif preprocessing == 'lemma':
			# 			preprocessed = preprocesser.lemmatize(word)
			# 		if preprocessed != word:
			# 			keywords_augmented.append(('preprocessed: ' + word + ' -> ' + preprocessed, (None, None), None, None))

			# Create the entity from words in n-gram
			if preprocessing in [None, 'None', 'none', 'null']:
				entity = ' '.join(nngram)
			if preprocessing == 'stem':
				entity = ' '.join([preprocesser.stem(word) for word in nngram])
			elif preprocessing == 'lemma':
				entity = ' '.join([preprocesser.lemmatize(word) for word in nngram])

			# Check the entity against the conditions provided
			try:
				if not entity in keywords and \
						sorted_ent_cls_idx[entity] is not None and \
						entity not in stopwords_nltk and \
						entity not in stopwords_custom \
						and entity.split()[-1] not in ['of']:
					# Calculate the span of the entity in the text
					start_index = text.find(entity)
					end_index = start_index + len(entity) if start_index != -1 else -1
					# Search for Wikidata QID and Wikipedia URL
					qid, url = get_wikidata_qid_wikipedia_url(entity)
					# Only store the entity if it has a Wikipedia URL
					if url is not None:
						# Store the entity (to prevent duplicates) and its augmentations (tupel)
						keywords.append(entity)
						#keywords_augmented.append((entity, (start_index, end_index), qid, url))
						keywords_augmented.append({'entity': entity, 'span': [start_index, end_index], 'qid': qid, 'link': url})
			except Exception as e:
				if verbose:
					print(f'Error: {e}')

	return keywords_augmented

def get_linked_text_html(text, preprocessing):

	keywords_augmented = get_keywords_spans_qids_wikip(text, preprocessing)

	linked_text_html = text

	for keyword in keywords_augmented:
		entity = keyword['entity']
		link = keyword['link']
		if link is not None:
			linked_text_html = linked_text_html.replace(entity, f'<a href="{link}">{entity}</a>')

	return linked_text_html

def get_mscs(keywords):
	mscs_predicted_single = []
	mscs_predicted_stat = {}

	for entity in keywords:
		try:
			mscs_predicted_single.extend(list(sorted_ent_cls_idx[entity])[0:1])
			for cls in sorted_ent_cls_idx[entity].items():
				try:
					# SELECTION HERE
					mscs_predicted_stat[
						cls[0]] += 1  # cls[1]#1 # weightedcontribution or binarycontribution
				except:
					mscs_predicted_stat[cls[0]] = 1
		except:
			pass

	# sort
	sorted_mscs_predicted_stat = dict(
		sorted(mscs_predicted_stat.items(), key=lambda item: item[1], reverse=True))

	return mscs_predicted_single, sorted_mscs_predicted_stat


def extract_keywords(text):

	print('Extracting keywords...')

	# get keywords
	keywords, qids = get_keywords(text, get_qids=True)

	if len(keywords) == 0:
		print('Could not extract keywords')
	else:
		try:
			print('Extracted keywords')
			# print keywords
			qids_url_prefix = 'https://www.wikidata.org/wiki/'  # e.g., Q11412
			qids_url_string = ''
			i = 0
			for keyword in keywords:
				try:
					qid = qids[i]
					if qid is not None:
						qids_url_string += f'<a href="{qids_url_prefix + qid}">{keyword}</a>, '
					else:
						qids_url_string += keyword + ', '
				except:
					qids_url_string += keyword + ', '
				# print(keyword)
				i += 1
			print(qids_url_string.rstrip(', '))
		except:
			print('Could not extract keywords')

	return keywords, qids


def extract_keywords_augmented(text, preprocessing):

	keywords_with_spans = get_keywords_spans_qids_wikip(text, preprocessing)

	return keywords_with_spans

def predict_mscs(keywords):

	print('Predicting MSCs...')

	# get mscs
	# keywords,qids = get_keywords(text,get_qids=True)
	mscs_predicted_single, sorted_mscs_predicted_stat = get_mscs(keywords)
	mscs = list(sorted_mscs_predicted_stat)[:nr_mscs_cutoff]  # mscs_predicted_single[:nr_mscs_cutoff]
	if len(mscs) == 0:
		print('Could not predict MSCs')
	else:
		try:
			print('Predicted MSCs')
			# print mscs
			# mscs_url_prefix = 'https://mathscinet.ams.org/mathscinet/msc/msc2010.html?t=' #e.g., 81V17
			mscs_url_prefix = 'https://zbmath.org/classification/?q=cc:'
			mscs_url_string = ''
			for msc in mscs:
				mscs_url_string += f'<a href="{mscs_url_prefix + msc}">{msc}</a>, '
				# print(msc)
			print(mscs_url_string.rstrip(', '))
		except:
			print('Could not predict MSCs')

	return mscs


# EXECUTE

# load stopwords
with open("./stopwords.txt", 'r') as f:
	stopwords_custom = f.readlines()

# load index
print('Loading indexes...')
sorted_ent_cls_idx = load_index()
print('Indexes loaded.')
