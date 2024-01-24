import streamlit as st
import json
import nltk
nltk.download('stopwords')
from nltk import ngrams
from nltk.corpus import stopwords as stopwords_nltk
stopwords_nltk = stopwords_nltk.words('english')
import pywikibot
from matplotlib import pyplot as plt

# DEFINE

# parameters
nr_mscs_cutoff = 5
nr_keywords_cutoff = 5


# from evaluate_classification import load_index
def load_index():

    # ent_cls_idx
    st.write('Loading entity-class index...')
    try:
        sorted_ent_cls_idx = st.session_state['ent_cls_idx']
    except:
        with open("./data/ent_cls_idx.json", 'r') as f:
            sorted_ent_cls_idx = json.load(f)
            st.session_state['ent_cls_idx'] = sorted_ent_cls_idx
    st.write('...done!')

    # cls_ent_idx
    st.write('Loading class-entity index...')
    try:
        sorted_cls_ent_idx = st.session_state['cls_ent_idx']
    except:
        with open("./data/cls_ent_idx.json", 'r') as f:
            sorted_cls_ent_idx = json.load(f)
            st.session_state['cls_ent_idx'] = sorted_cls_ent_idx
    st.write('...done!')

    return sorted_cls_ent_idx, sorted_ent_cls_idx


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
        stopwords_custom = [stopword.strip() for stopword in f.readlines()]

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
                    if sorted_ent_cls_idx[entity] is not None and\
                            entity not in stopwords_nltk and\
                            entity not in stopwords_custom and entity.split()[-1] not in ['of']:
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

    st.write('Extracting keywords...')

    # get keywords
    keywords, qids = get_keywords(text, get_qids=True)
    if len(keywords) == 0:
        st.error('Could not extract keywords')
    else:
        try:
            #st.success('Extracted keywords')
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
                # st.write(keyword)
                i += 1
            st.markdown(qids_url_string.rstrip(', '), unsafe_allow_html=True)
            st.session_state['keywords'] = keywords
        except:
            st.error('Could not extract keywords')

    return keywords


def predict_mscs(keywords):

    st.write('Predicting MSCs...')

    # get mscs
    # keywords,qids = get_keywords(text,get_qids=True)
    mscs_predicted_single, sorted_mscs_predicted_stat = get_mscs(keywords)
    mscs = list(sorted_mscs_predicted_stat)[:nr_mscs_cutoff]  # mscs_predicted_single[:nr_mscs_cutoff]
    if len(mscs) == 0:
        st.error('Could not predict MSCs')
    else:
        try:
            #st.success('Predicted MSCs')
            # print mscs
            # mscs_url_prefix = 'https://mathscinet.ams.org/mathscinet/msc/msc2010.html?t=' #e.g., 81V17
            mscs_url_prefix = 'https://zbmath.org/classification/?q=cc:'
            mscs_url_string = ''
            for msc in mscs:
                mscs_url_string += f'<a href="{mscs_url_prefix + msc}">{msc}</a>, '
                # st.write(msc)
            st.markdown(mscs_url_string.rstrip(', '), unsafe_allow_html=True)
            st.session_state['mscs'] = mscs
        except:
            st.error('Could not predict MSCs')

    return mscs


def plot_msc_keywords_distribution(mscs, keywords, key_prefix):

    explain_mode = st.selectbox('Explain', ['MSCs', 'Keywords'], key=key_prefix + '_mode')

    if explain_mode == 'MSCs':

        try:

            msc_keywords = {}
            for msc in mscs:
                try:
                    msc_keywords[msc] = global_sorted_cls_ent_idx[msc]
                except:
                    st.error('Could not explain MSC ' + msc)

            msc_selected = st.selectbox('Select MSC', mscs, key=key_prefix + '_MSC')

            predictions = list(msc_keywords[msc_selected].keys())[:nr_keywords_cutoff]
            predictions = [prediction.replace('nan', 'other') for prediction in predictions]
            confidences = list(msc_keywords[msc_selected].values())[:nr_keywords_cutoff]

        except:
            st.error('Could not explain MSCs')

    elif explain_mode == 'Keywords':

        try:

            keyword_mscs = {}
            for keyword in keywords:
                try:
                    keyword_mscs[keyword] = sorted_ent_cls_idx[keyword]
                except:
                    st.error('Could not explain keyword ' + keyword)

            keyword_selected = st.selectbox('Select keyword', keywords, key=key_prefix + '_keyword')

            predictions = list(keyword_mscs[keyword_selected].keys())[:nr_mscs_cutoff]
            predictions = [prediction.replace('nan', 'other') for prediction in predictions]
            confidences = list(keyword_mscs[keyword_selected].values())[:nr_mscs_cutoff]

        except:
            st.error('Could not explain keywords')

    # plot

    try:

        # Pie chart, where the slices will be ordered and plotted counter-clockwise:
        labels = predictions
        sizes = confidences
        # explode = (0, 0.1, 0, 0, 0)  # only "explode" the 2nd slice

        fig, ax = plt.subplots()
        ax.pie(sizes, labels=labels, autopct='%1.1f%%',
               shadow=True, startangle=90)
        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # plt.show()
        st.pyplot(fig)

    except:
        st.error('Could not plot prediction explanation')


# EXECUTE

# Markdown

st.set_page_config(page_title="zbMATH Fine-Grained MSC Prediction Explainer",
				   page_icon="./favicon_zbMATH.png")

hline = '''
---
'''
fline = f'<hr style="border: 4px solid gray; margin-top: 0.5em; margin-bottom: 0.5em;">'

hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.image('./Logo_zbMATH-Open-201223.svg', width=300)
header_text = 'Fine-Grained MSC Prediction Explainer'
st.header(header_text)

st.markdown(hline)

st.caption('INDEXES')

# load index
print('Loading indexes...')
st.write('Loading indexes...')
st.write('This may take around 1 minute to load.')
st.write('The indexes (json) have a combined size of around 1GB.')
st.markdown("[In the meantime, please check out the interactive notebook for explanations](https://purl.org/fine-class)")
global_sorted_cls_ent_idx, sorted_ent_cls_idx = load_index()
st.success('Indexes loaded')

st.markdown(fline, unsafe_allow_html=True)
st.caption('MSCS FROM TEXT')

# show abstract
# from https://zbmath.org/?q=an%3A7525220
test_text = 'Summary: This paper presents a novel path integral formalism for Einstein’s theory of gravitation from ' \
            'the viewpoint of optimal control theory. Despite its close connection to the well-known variational ' \
            'principle of physicists, optimal control turns out to be more general. Within this context, a Lagrangian ' \
            'which is different from the Einstein-Hilbert Lagrangian is defined. Einstein’s field equations are ' \
            'recovered exactly with variations of the new action functional. The quantum theory is obtained using ' \
            'Ashtekar variables and the loop scalar product. As an illustrative example, the tunneling process of a ' \
            'black hole into another black hole or into a white hole is investigated with a toy model. '
text = st.text_area('Document abstract', value=test_text)

st.markdown('[You can copy more example abstracts from zbMATH Open...](https://zbmath.org)')
st.write('...or enter your own mathematical document abstract')

st.write(hline)

# if st.button('Extract keywords and predict MSCs'):

st.session_state['keywords'] = []
st.session_state['mscs'] = []

# if st.button('Extract keywords'):
predicted_keywords = extract_keywords(text)

st.write(hline)

# if st.button('Predict MSCs'):
text_mscs = predict_mscs(predicted_keywords)

plot_msc_keywords_distribution(text_mscs, predicted_keywords, key_prefix='predicted_keywords')

st.markdown(fline, unsafe_allow_html=True)
st.caption('MSCS FROM KEYWORDS')

# enter keywords
# from https://zbmath.org/?q=an%3A7525220
st.caption('Enter custom keywords (; separated)')
#test_keywords = 'loop quantum gravity; optimal control; path integral; transition of geometry'
test_keywords = '; '.join(predicted_keywords)
custom_keywords = st.text_input('Document keywords', value=test_keywords)

st.write(hline)

# if st.button('Predict MSCs'):
custom_keywords = [kw.lstrip().rstrip() for kw in custom_keywords.split(';')]
total_keyword_mscs = predict_mscs(custom_keywords)

plot_msc_keywords_distribution(total_keyword_mscs, custom_keywords, key_prefix='custom_keywords')

# TODO: MSC/QID tree
