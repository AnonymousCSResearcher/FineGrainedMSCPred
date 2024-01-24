import json
import nltk
nltk.download('stopwords')
from nltk import ngrams
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
    with open("./data/ent_cls_idx.json", 'r') as f:
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


def get_keywords(text, get_qids):
    with open("./stopwords.txt", 'r') as f:
        stopwords_custom = f.readlines()

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


# load index
print('Loading indexes...')
sorted_ent_cls_idx = load_index()
print('Indexes loaded.')
