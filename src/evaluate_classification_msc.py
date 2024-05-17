import pandas as pd
import json

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import WordNetLemmatizer
from nltk import ngrams

from collections import Counter
import math
import numpy as np
import scipy

from matplotlib import pyplot as plt

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_stopwords():
    with open("../stopwords.txt", 'r') as f:
        sstopwords = f.readlines()
    return sstopwords


def remove_stopwords_custom(text):
    # remove stopwords and punctuation
    text = text.lower()
    sstopwords = get_stopwords()
    punctuations = [',', ';', '.', '!', '?', ':', '(', ')', '\\', '+', '-']
    for stopword in sstopwords:
        text = text.replace(stopword, "")
    for punctuation in punctuations:
        text = text.replace(punctuation, "")
    return text


def remove_stopwords_nltk(text):
    text_tokens = word_tokenize(text)
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words()]
    return tokens_without_sw


def stemming_lemmatization(word):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word)


def nlp_clean(text):
    text = remove_stopwords_custom(text)
    tokens_without_sw = remove_stopwords_nltk(text)
    clean_words = []
    for word in tokens_without_sw:
        clean_words.append(stemming_lemmatization(word))
    return ' '.join(clean_words)


def get_anchor(qid, name):
    linked = qid + ">" + name
    anchor = """<a href="https://www.wikidata.org/wiki/%s</a>""" % linked
    return anchor


file = "enwiki-latest-all-titles-in-ns0"


def get_Wikipedia_article_names(n_gram_length):
    file = "enwiki-latest-all-titles-in-ns0"

    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # names = []
    names = {}
    for line in lines:
        if len(line.split("_")) == n_gram_length:
            name = line.strip("\n").replace("_", " ").replace('"', '').lower()
            # LIST
            # names.append(name)
            # DICT
            names[name] = name

    return names


def get_entity_linking_wikipedia(text, n_gram_length):
    # load Wikipedia article name candidates
    Wikipedia_article_names = get_Wikipedia_article_names(n_gram_length)

    nngrams = ngrams(text.split(), n=n_gram_length)

    # get entity candidates
    entities = []
    for nngram in nngrams:
        name = ''
        for word in nngram:
            name += word + " "
        print(name)
        try:
            entities.append(Wikipedia_article_names[name[:-1]])
        except:
            pass

    return entities


def clean(string):
    if not isinstance(string, str):
        return ""
    return string.replace('[', '').replace(']', '').replace('\\', '').replace("'", '')


def get_mscs(table, idx):
    mscs = []
    for msc in clean(table['msc'][idx]).split():
        msc = msc.strip(',')
        mscs.append(msc)
    return mscs


def get_keywords(table, idx):
    keywords = []
    for keyword in clean(str(table['keyword'][idx])).split(","):
        keyword = keyword.lstrip().rstrip()
        for clea_str in [',', "'", '"', "`", '\\']:
            keyword = keyword.strip(clea_str)
        keywords.append(keyword)
    return keywords


def get_references(table, idx):
    references = []
    for reference in clean(str(table['refs'][idx])).split(","):
        reference = reference.lstrip().rstrip()
        for clea_str in [',', "'", '"', "`", '\\']:
            reference = reference.strip(clea_str)
        references.append(reference)
    return references


def get_refs(table, idx):
    return clean(table['refs'][idx]).replace(',', '').split()


def get_de(table, idx):
    return table['de'][idx]


def load_indexes():

    indexes = {}
    index_files = ['cls_ent_idx.json', 'ent_cls_idx.json',
                    'cls_ref_idx.json', 'ref_cls_idx.json']
    for index_file in index_files:
        print('Loading index ' + index_file)
        try:
            with open('../data/' + index_file, 'r') as f:
                indexes[index_file] = json.load(f)
        except Exception as e:
            print(e)
            print('Generating index ' + index_file)
            import generate_indexes

    return indexes


def predict_text_mscs(table,indexes):
    # create prediction table
    prediction_table = pd.DataFrame(columns=['de', 'mscs_actual', 'mscs_predicted', 'confidences', 'overlap_ratio'])

    # open index
    #sorted_cls_ent_idx = indexes['cls_ent_idx.json']
    sorted_ent_cls_idx = indexes['ent_cls_idx.json']

    # mscs actual vs. predicted
    mscs_actual = {}
    mscs_predicted = {}
    mscs_pred_conf = {}
    overlap_ratios = []

    sstopwords = get_stopwords()
    # predict mscs for each doc abs text
    tot_rows = len(table)
    latest_progress = 0
    for idx in range(tot_rows):
        # print(idx)
        current_progress = round(idx / tot_rows * 100, 1)
        if current_progress != latest_progress and current_progress % 10 == 0:
            print(current_progress, '%')
            latest_progress = current_progress
        de = table['de'][idx]
        # if de in mrmscs:
        text = table['text'][idx]
        mscs_actual[idx] = get_mscs(table, idx)
        mscs_predicted_stat = {}
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
                        if sorted_ent_cls_idx[entity] is not None and entity not in sstopwords:
                            # print(entity)
                            # mscs_predicted[idx].extend(list(sorted_ent_cls_idx[entity])[0:1])
                            for cls in sorted_ent_cls_idx[entity].items():
                                try:
                                    # SELECTION HERE
                                    mscs_predicted_stat[
                                        cls[0]] += 1  # cls[1]#1 # weightedcontribution or binarycontribution
                                except:
                                    mscs_predicted_stat[cls[0]] = 1
                    except:
                        pass
            except:
                pass

        if len(mscs_predicted_stat) != 0:
            # sort
            sorted_mscs_predicted_stat = dict(
                sorted(mscs_predicted_stat.items(), key=lambda item: item[1], reverse=True))

            # get (normalized) prediction (confidence)
            mscs_predicted[idx] = list(sorted_mscs_predicted_stat)[
                                  :nr_mscs_cutoff]  # cut off at fixed nr_mscs_cutoff or dynamic number #len(mscs_actual[idx])
            mscs_cut_off = list(sorted_mscs_predicted_stat.items())[:nr_mscs_cutoff]
            # norm
            tot = sum(v for k, v in mscs_cut_off)
            # confidence
            mscs_pred_conf[idx] = [v / tot for k, v in mscs_cut_off]

            # compare mscs actual to predicted
            common_mscs = len(set(mscs_actual[idx]).intersection(set(mscs_predicted[idx])))
            total_mscs = nr_mscs_cutoff  # len(mscs_predicted[idx])# + len(mscs_actual[idx])
            overlap_ratio = round(common_mscs / total_mscs, 3)
            overlap_ratios.append(overlap_ratio)
            # print(idx,overlap_ratio)

            # extend prediction table
            new_row = {'de': table['de'][idx], 'mscs_actual': mscs_actual[idx],
                       'mscs_predicted': mscs_predicted[idx],
                       'confidences': mscs_pred_conf[idx], 'overlap_ratio': overlap_ratio}
            prediction_table = prediction_table.append(new_row, ignore_index=True)
            pass

    # save prediction table
    print('save prediction table...')
    prediction_table.to_csv('../data/mscs_prediction_table_text_cutoff' + str(nr_mscs_cutoff) + '.csv')

    #print('Overlap: ' + str(np.mean(overlap_ratios)))

    return None


def predict_keyword_mscs(table,indexes):
    # create prediction table
    prediction_table = pd.DataFrame(columns=['de', 'mscs_actual', 'mscs_predicted', 'confidences', 'overlap_ratio'])

    # open index
    #sorted_cls_ent_idx = indexes['cls_ent_idx.json']
    sorted_ent_cls_idx = indexes['ent_cls_idx.json']

    # mscs actual vs. predicted
    mscs_actual = {}
    mscs_predicted = {}
    mscs_pred_conf = {}
    overlap_ratios = []

    # predict mscs for each doc keywords
    tot_rows = len(table)
    latest_progress = 0
    for idx in range(tot_rows):
        # print(idx)
        current_progress = round(idx / tot_rows * 100, 1)
        if current_progress != latest_progress and current_progress % 10 == 0:
            print(current_progress, '%')
            latest_progress = current_progress
        de = table['de'][idx]
        # if de in mrmscs:
        keywords = get_keywords(table, idx)
        mscs_actual[idx] = get_mscs(table, idx)

        mscs_predicted_stat = {}

        for keyword in keywords:
            try:
                for cls in sorted_ent_cls_idx[keyword].items():
                    try:
                        # SELECTION HERE
                        mscs_predicted_stat[
                            cls[0]] += 1  # cls[1]#1 # weightedcontribution or binarycontribution
                    except:
                        mscs_predicted_stat[cls[0]] = 1
            except:
                pass

        if len(mscs_predicted_stat) != 0:
            # sort
            sorted_mscs_predicted_stat = dict(
                sorted(mscs_predicted_stat.items(), key=lambda item: item[1], reverse=True))

            # get (normalized) prediction (confidence)
            mscs_predicted[idx] = list(sorted_mscs_predicted_stat)[
                                  :nr_mscs_cutoff]  # cut off at fixed nr_mscs_cutoff or dynamic number #len(mscs_actual[idx])
            mscs_cut_off = list(sorted_mscs_predicted_stat.items())[:nr_mscs_cutoff]
            # norm
            tot = sum(v for k, v in mscs_cut_off)
            # confidence
            mscs_pred_conf[idx] = [v / tot for k, v in mscs_cut_off]

            # compare mscs actual to predicted
            common_mscs = len(set(mscs_actual[idx]).intersection(set(mscs_predicted[idx])))
            total_mscs = nr_mscs_cutoff  # len(mscs_predicted[idx])# + len(mscs_actual[idx])
            overlap_ratio = round(common_mscs / total_mscs, 3)
            overlap_ratios.append(overlap_ratio)
            # print(idx,overlap_ratio)

        # extend prediction table
        new_row = {'de': table['de'][idx], 'mscs_actual': mscs_actual[idx],
                   'mscs_predicted': mscs_predicted[idx],
                   'confidences': mscs_pred_conf[idx], 'overlap_ratio': overlap_ratio}
        prediction_table = prediction_table.append(new_row, ignore_index=True)
        pass

    # save prediction table
    print('save prediction table...')
    prediction_table.to_csv('../data/mscs_prediction_table_keywords_cutoff' + str(nr_mscs_cutoff) + '.csv')


def predict_reference_mscs(table,indexes):
    # create prediction table
    prediction_table = pd.DataFrame(columns=['de', 'mscs_actual', 'mscs_predicted', 'confidences', 'overlap_ratio'])

    # open index
    #sorted_cls_ref_idx = indexes['cls_ref_idx.json']
    sorted_ref_cls_idx = indexes['ref_cls_idx.json']

    # mscs actual vs. predicted
    mscs_actual = {}
    mscs_predicted = {}
    mscs_pred_conf = {}
    overlap_ratios = []

    # predict mscs for each doc keywords
    tot_rows = len(table)
    latest_progress = 0
    for idx in range(tot_rows):
        # print(idx)
        current_progress = round(idx / tot_rows * 100, 1)
        if current_progress != latest_progress and current_progress % 10 == 0:
            print(current_progress, '%')
            latest_progress = current_progress
        de = table['de'][idx]
        # if de in mrmscs:
        references = get_references(table, idx)
        mscs_actual[idx] = get_mscs(table, idx)

        mscs_predicted_stat = {}

        for reference in references:
            try:
                for cls in sorted_ref_cls_idx[reference].items():
                    try:
                        # SELECTION HERE
                        mscs_predicted_stat[
                            cls[0]] += 1  # cls[1]#1 # weightedcontribution or binarycontribution
                    except:
                        mscs_predicted_stat[cls[0]] = 1
            except:
                pass

        if len(mscs_predicted_stat) != 0:
            # sort
            sorted_mscs_predicted_stat = dict(
                sorted(mscs_predicted_stat.items(), key=lambda item: item[1], reverse=True))

            # get (normalized) prediction (confidence)
            mscs_predicted[idx] = list(sorted_mscs_predicted_stat)[
                                  :nr_mscs_cutoff]  # cut off at fixed nr_mscs_cutoff or dynamic number #len(mscs_actual[idx])
            mscs_cut_off = list(sorted_mscs_predicted_stat.items())[:nr_mscs_cutoff]
            # norm
            tot = sum(v for k, v in mscs_cut_off)
            # confidence
            mscs_pred_conf[idx] = [v / tot for k, v in mscs_cut_off]

            # compare mscs actual to predicted
            common_mscs = len(set(mscs_actual[idx]).intersection(set(mscs_predicted[idx])))
            total_mscs = nr_mscs_cutoff  # len(mscs_predicted[idx])# + len(mscs_actual[idx])
            overlap_ratio = round(common_mscs / total_mscs, 3)
            overlap_ratios.append(overlap_ratio)
            # print(idx,overlap_ratio)

            # extend prediction table
            new_row = {'de': table['de'][idx], 'mscs_actual': mscs_actual[idx],
                       'mscs_predicted': mscs_predicted[idx],
                       'confidences': mscs_pred_conf[idx], 'overlap_ratio': overlap_ratio}
            prediction_table = prediction_table.append(new_row, ignore_index=True)
            pass

    # save prediction table
    print('save prediction table...')
    prediction_table.to_csv('../data/mscs_prediction_table_references_cutoff' + str(nr_mscs_cutoff) + '.csv')


def get_sparse_mscs(table):
    # Create msc frequency index
    msc_freq_idx = {}

    # Iterate documents to get index
    num_rows = len(table)
    for idx in range(num_rows):
        mscs = table['msc'][idx].split()
        for msc in mscs:
            try:
                msc_freq_idx[msc] += 1
            except:
                msc_freq_idx[msc] = 1

    # Get sparse mscs
    for msc in msc_freq_idx.items():
        msc_name = msc[0]
        msc_freq = msc[1]
        if msc_freq < 20:
            print(msc_name)
    # 1003 results for < 10
    # 1507 results for < 10


def predict_mscs(ent_cls_dict):
    # get confidences
    for ent in ent_cls_dict.items():
        ent_key = ent[0]
        ent_val = ent[1]
        total = sum(ent_val.values(), 0.0)
        for msc in ent_val.items():
            msc_key = msc[0]
            msc_val = msc[1]
            ent_cls_dict[ent_key][msc_key] /= total


def print_index_statistics(indexes):

    for index in indexes.items():

        index_name = index[0]
        index_cont = index[1]

        # average length
        def get_mean_count(idx):
            lengths = []
            for item in idx.items():
                lengths.append(len(item[1]))
            avg_length = np.mean(lengths)
            return avg_length

        print('\nStats of index ' + index_name)

        idx_avg_count = get_mean_count(index_cont)
        print('Average entry per key count: ' + str(idx_avg_count))

        # entropy
        def get_mean_entropy(idx):
            entropies = []
            for cls in idx.items():
                frequencies = [ent[1] for ent in cls[1].items()]
                entropies.append(scipy.stats.entropy(frequencies))
            avg_entropy = np.mean(entropies)
            return avg_entropy

        idx_avg_entropy = get_mean_entropy(index_cont)
        print('Average entry per key entropy: ' + str(idx_avg_entropy))

    print('\n')

    return 0


def load_data():
    # load data
    # msc(keyword) index
    with open(dict_path, 'r') as f:
        keyword_msc_index = json.load(f)
    # raw data
    raw_data = None  # pd.read_csv(data_path)

    return raw_data, keyword_msc_index


def get_mrmscs_dict():
    with open('../data/mrmscs_dict.json', 'r') as f:
        return json.load(f)


def get_dcg(actual_mscs, predicted_mscs):
    i_max = len(actual_mscs)
    j_max = len(predicted_mscs)

    dcgs = []
    for i in range(i_max):
        msc_actual = actual_mscs[i]
        dcg = 0
        for j in range(j_max):
            msc_predicted = predicted_mscs[j]
            if msc_actual == msc_predicted:
                # score and rank
                if i == 1:
                    score = 2
                else:
                    score = 1
                rank = j + 1
                # DCG
                dcg += score / math.log2(rank + 1)
        dcgs.append(dcg)

    # average over actual mscs
    dcg = 0
    if len(dcgs) != 0:
        dcg = np.mean(dcgs)

    return dcg


def get_dcg_table(raw_data, mrmscs_dict, keyword_msc_index):
    # predict and evaluate
    row_list = []
    num_rows = len(raw_data)
    na_counter = 0
    filtered = raw_data[raw_data['de'].isin(
        list(map(int, mrmscs_dict.keys())))
    ]
    filtered.set_index('de').to_csv('../data/out-mr.csv')
    for row in filtered.itertuples():
        try:
            mrmscs = mrmscs_dict[str(row.de)][:nr_mscs_cutoff]
        except KeyError:
            na_counter += 1
            continue
        mscs = []
        for msc in clean(row.msc).split():
            msc = msc.strip(',')
            mscs.append(msc)
        # proceed only if mscs and mrmscs available
        if len(mscs) == 0 or len(mrmscs) == 0:
            continue

        # get keyword mscs
        keywords = []
        for keyword in clean(str(row.keyword)).split(","):
            keyword = keyword.lstrip().rstrip()
            for clea_str in [',', "'", '"', "`", '\\']:
                keyword = keyword.strip(clea_str)
            keywords.append(keyword)
        keywords_mscs = []
        for keyword in keywords:
            try:
                keywords_mscs.extend(keyword_msc_index[keyword])
            except:
                pass
        keywords_mscs = list(Counter(keywords_mscs[:nr_mscs_cutoff]))

        # get reference mscs
        refs = clean(row.refs).replace(',', '').split()
        refs_mscs = list(Counter(refs))[:nr_mscs_cutoff]

        # get intersection and union of keyword and reference mscs
        keyword_and_refs_mscs = list(set(keywords_mscs).intersection(set(refs_mscs)))
        keyword_or_refs_mscs = list(set(keywords_mscs).union(set(refs_mscs)))

        # populate evaluation table
        # get nDCGs
        # ideal DCG for normalization
        IDCG = get_dcg(mscs, mscs)
        # other nDCGs (mrmscs, keywords, refs)
        nDCG_mrmscs = get_dcg(mscs, mrmscs) / IDCG
        nDCG_keywords = get_dcg(mscs, keywords_mscs) / IDCG
        nDCG_refs = get_dcg(mscs, refs_mscs) / IDCG
        nDCG_keywords_and_refs = get_dcg(mscs, keyword_and_refs_mscs) / IDCG
        nDCG_keywords_or_refs = get_dcg(mscs, keyword_or_refs_mscs) / IDCG

        # append and save evaluation table
        new_row = {'de': row.de, 'mscs': mscs, 'mrmscs': mrmscs,
                   'keyword_mscs': keywords_mscs,
                   'refs_mscs': refs_mscs,
                   'nDCG_mrmscs': nDCG_mrmscs,
                   'nDCG_keywords': nDCG_keywords,
                   'nDCG_refs': nDCG_refs,
                   'nDCG_keywords_and_refs': nDCG_keywords_and_refs,
                   'nDCG_keywords_or_refs': nDCG_keywords_or_refs}
        row_list.append(new_row)

    print('Matching mscs/mrmscs: ' + str((1 - na_counter / num_rows) * 100) + '%')

    # save
    eval_table = pd.DataFrame(row_list)
    eval_table.to_csv(base_path)


def compare_DCGs(eval_table):
    if eval_table.empty:
        print('nothing to evaluate')
        return
    # mrmscs
    list_nDCG_mrmscs = list(eval_table['nDCG_mrmscs'])
    mean_nDCG_mrmscs = np.mean(list_nDCG_mrmscs)
    print('mean_nDCG_zbmath: ' + str(mean_nDCG_mrmscs))

    # keywords
    list_nDCG_keywords = list(eval_table['nDCG_keywords'])
    mean_nDCG_keywords = np.mean(list_nDCG_keywords)
    print('mean_nDCG_keywords: ' + str(mean_nDCG_keywords))

    # refs
    list_nDCG_refs = list(eval_table['nDCG_refs'])
    mean_nDCG_refs = np.mean(list_nDCG_refs)
    print('mean_nDCG_refs: ' + str(mean_nDCG_refs))

    # keywords AND refs
    list_nDCG_keywords_and_refs = list(eval_table['nDCG_keywords_and_refs'])
    mean_nDCG_keywords_and_refs = np.mean(list_nDCG_keywords_and_refs)
    print('mean_nDCG_keywords_and_refs: ' + str(mean_nDCG_keywords_and_refs))

    # keywords OR refs
    list_nDCG_keywords_or_refs = list(eval_table['nDCG_keywords_or_refs'])
    mean_nDCG_keywords_or_refs = np.mean(list_nDCG_keywords_or_refs)
    print('mean_nDCG_keywords_or_refs: ' + str(mean_nDCG_keywords_or_refs))


def compare_mr_keyword_refs_dcgs(raw_data):
    # get eval table
    print('Load data')
    _, keyword_msc_index = load_data()
    mrmscs_dict = get_mrmscs_dict()
    print('Get DCGs')
    get_dcg_table(raw_data, mrmscs_dict, keyword_msc_index)

    # eval eval table
    print('Score eval')
    eval_table = pd.read_csv(base_path)
    compare_DCGs(eval_table)


def get_precision_recall_curves():
    # load prediction tables
    prediction_table_text = pd.read_csv(
        '../data/mscs_prediction_table_text_cutoff' + str(nr_mscs_cutoff) + '.csv')
    prediction_table_keywords = pd.read_csv(
        '../data/mscs_prediction_table_keywords_cutoff' + str(nr_mscs_cutoff) + '.csv')
    prediction_table_references = pd.read_csv(
        '../data/mscs_prediction_table_references_cutoff' + str(nr_mscs_cutoff) + '.csv')

    # load mrmscs baseline
    with open('../data/mrmscs_dict.json', 'r') as f:
        mrmscs_dict = json.load(f)

    # evaluate predictions
    # ir measures
    # baseline = mscs (mr)
    # competing origins = predicted vs. zbmath
    mscs_origins = ['predicted_from_text', 'predicted_from_keywords', 'predicted_from_references', 'human_baseline']
    precision_recall = {mscs_origins[0]: {}, mscs_origins[1]: {}, mscs_origins[2]: {}, mscs_origins[3]: {}}

    # len(prediction_table_text) < len(prediction_table_keywords) : text not always available

    latest_progress = 0
    for idx in range(len(prediction_table_text)):
        #print(idx / len(prediction_table_text))
        current_progress = round(idx / len(prediction_table_text) * 100, 1)
        if current_progress != latest_progress and current_progress % 10 == 0:
            print(current_progress, '%')
            latest_progress = current_progress

        # collect mscs
        mscs_dict = {}

        # document nr.
        de = prediction_table_text['de'][idx]

        # mscs (mr)
        mscs_dict['mscs_mr'] = mrmscs_dict[str(de)]

        # mscs (zbmath) = competitor
        mscs_zbmath = prediction_table_text['mscs_actual'][idx]
        mscs_dict['human_baseline'] = mscs_zbmath.replace("'", "").lstrip('[').rstrip(']').strip().split(', ')

        # mscs (predicted_text) = competitor
        mscs_predicted_text = prediction_table_text['mscs_predicted'][idx]
        mscs_dict['predicted_from_text'] = mscs_predicted_text.replace("'", "").lstrip('[').rstrip(']').strip().split(
            ', ')

        # mscs (predicted_keywords) = competitor
        idxx = prediction_table_keywords.index[prediction_table_keywords['de'] == de]
        idxx = idxx.tolist()[0]
        # idxx != idx because prediction tables are not in same order
        mscs_predicted_keywords = prediction_table_keywords['mscs_predicted'][idxx]
        mscs_dict['predicted_from_keywords'] = mscs_predicted_keywords.replace("'", "").lstrip('[').rstrip(
            ']').strip().split(', ')

        # mscs (predicted_references) = competitor
        idxx = prediction_table_references.index[prediction_table_references['de'] == de]
        try:
            idxx = idxx.tolist()[0]
            # idxx != idx because prediction tables are not in same order
            mscs_predicted_references = prediction_table_references['mscs_predicted'][idxx]
            mscs_dict['predicted_from_references'] = mscs_predicted_references.replace("'", "").lstrip('[').rstrip(
                ']').strip().split(', ')

            # mscs (actual) = baseline
            mscs_actual = mscs_dict['mscs_mr']

            for mscs_origin in mscs_origins:

                mscs_predicted_full = mscs_dict[mscs_origin]

                for i in range(1, nr_mscs_cutoff + 1):
                    mscs_predicted = mscs_predicted_full[:i]
                    # https://stats.stackexchange.com/questions/21551/how-to-compute-precision-recall-for-multiclass-multilabel-classification
                    # precision = ratio of how much of the predicted is correct
                    mscs_intersection = [msc for msc in mscs_predicted if msc in mscs_actual]
                    if len(mscs_predicted) > 0:
                        precision = len(mscs_intersection) / len(mscs_predicted)
                    else:
                        precision = 1
                    # recall = ratio of how many of the actual labels were predicted
                    if len(mscs_actual) > 0:
                        recall = len(mscs_intersection) / len(mscs_actual)
                    else:
                        recall = 1

                    try:
                        precision_recall[mscs_origin][i]['precisions'].append(precision)
                        precision_recall[mscs_origin][i]['recalls'].append(recall)
                    except:
                        try:
                            precision_recall[mscs_origin][i]['precisions'] = [precision]
                            precision_recall[mscs_origin][i]['recalls'] = [recall]
                        except:
                            precision_recall[mscs_origin][i] = {'precisions': [precision]}
                            precision_recall[mscs_origin][i] = {'recalls': [recall]}

        except:
            pass

    # plot metrics
    # precision-recall curve
    fig, ax = plt.subplots()
    # collect metrics for plot
    for mscs_origin in mscs_origins:

        pr_rc = precision_recall[mscs_origin]
        precisions = []
        recalls = []
        cutoffs = []
        for p_r in pr_rc.items():
            cutoffs.append(p_r[0])
            precisions.append(np.mean(p_r[1]['precisions']))
            recalls.append(np.mean(p_r[1]['recalls']))

        marker_dict = {mscs_origins[0]: 'x', mscs_origins[1]: 's', mscs_origins[2]: 'o', mscs_origins[3]: '>'}
        ax.scatter(recalls, precisions, marker=marker_dict[mscs_origin], s=10, label=mscs_origin)

    # Center the legend
    legend = ax.legend(loc='center', bbox_to_anchor=(0.5, 0.5), fontsize=12)
    # Ensure the legend box has a background and frame
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('black')

    # # set plot parameters
    # plt.xlim(0, 0.8)
    plt.ylim(0, 0.75)

    # set plot labels
    plt.title('Precision-Recall Curve for MSC Prediction')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.savefig('../data/prec-rec-curve_mscs.pdf', format="pdf", bbox_inches="tight")
    plt.show()


###########
# EXECUTE #
###########

# Set paths
data_path = '../data/out.csv'  # full dataset
dict_path = '../data/ent_cls_idx.json' # index
test_path = '../data/out-mr.csv'  # test selection
base_path = '../data/keywords_vs_refs_mrmscs.csv'
mrms_path = '../data/msc-zbmath-mr.csv'

# Set parameters
nr_mscs_cutoff = 10
only_plot = True

# Execute pipeline
if not only_plot:

    # 1) Load train table
    print('\nLoad train table...\n')
    train_table = pd.read_csv(data_path, delimiter=',')

    # Set parameter
    tot_rows = len(train_table)
    nr_docs = int(tot_rows)

    # 2) Get mapping indexes
    print('\nGet mapping indexes...\n')
    indexes = load_indexes()

    # 3) Index statistics
    print('\nIndex statistics:\n')
    print_index_statistics(indexes)

    # 4) Load test table
    print('\nLoad test table...\n')
    test_table = pd.read_csv(test_path, delimiter=',')

    # 5) Predict MSCs
    print('\nPredict MSCs...\n')
    print('Predict from text...')
    predict_text_mscs(test_table,indexes)
    print('Predict from keywords...')
    predict_keyword_mscs(test_table,indexes)
    print('Predict from references...')
    predict_reference_mscs(test_table,indexes)

    # 6) Evaluate MSC predictions
    print('\nEvaluate MSC predictions...\n')
    # Compare to MR-MSCs and References-MSCs
    compare_mr_keyword_refs_dcgs(test_table)

# 7) Get precision-recall curves
print('\nGet precision-recall curves...')
get_precision_recall_curves()

# 8) Run MSC prediction explainer demo
print('\nRun MSC prediction explainer demo...\n')
# import run_viewer

print('end')
