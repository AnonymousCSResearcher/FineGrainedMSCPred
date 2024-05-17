import os
import json
import jsonlines
import nltk
from nltk import ngrams
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define paths
# root_path = os.path.dirname(os.getcwd())
# data_path = root_path + '/data'
# zbmed_path = data_path + '/ZB MED'
zbmed_path = 'D:\\ZB MED'
zbmed_file = 'test_50070.json' # 'keexport_simple_format.json' # 'ke_export_202404050852.jsonl'
json_lines_path = zbmed_path + '/' + zbmed_file


def load_dataset():

	# Assuming json_objects contains your JSON lines data
	json_lines = []
	with open(json_lines_path, 'r', encoding='utf-8') as f:
		for line in f:
			json_lines.append(json.loads(line))

	return json_lines


def load_indexes():

	# Load indexes from JSON files
	print('Load the ent_cls_index...')
	with open(zbmed_path + '/ent_cls_index.json', 'r', encoding='utf-8') as f:
		sorted_ent_cls_index = json.load(f)
	# print('Load the cls_ent_index...')
	# with open(zbmed_path + '/cls_ent_index.json', 'r', encoding='utf-8') as f:
	# 	sorted_cls_ent_index = json.load(f)
	sorted_cls_ent_index = None

	indexes = {'sorted_ent_cls_index': sorted_ent_cls_index,
				'sorted_cls_ent_index': sorted_cls_ent_index}

	return indexes

def load_stopwords_custom():
	# load stopwords
	with open("../stopwords.txt", 'r') as f:
		stopwords_custom = f.readlines()
	return stopwords_custom


# Load the NLTK stopwords
nltk.download('stopwords')
stopwords_nltk = stopwords.words('english')
# Load the custom stopwords
stopwords_custom = load_stopwords_custom()


def get_keywords(text, indexes):

	keywords = []

	n_gram_lengths = [1, 3]
	for n in n_gram_lengths:
		try:
			nngrams = ngrams(text.split(), n)
			for nngram in nngrams:
				entity = ''
				for word in nngram:
					entity += word + ' '
				entity = entity[:-1]
				try:
					#print(entity.title())
					# indexes['sorted_ent_cls_idx'][entity.title()] is not None and
					if entity.lower() not in stopwords_nltk and\
							entity.lower() not in stopwords_custom:
						keywords.append(entity.title())
				except:
					pass
		except:
			pass

	keywords = list(set(keywords))

	return keywords


def get_mesh_counts_from_keywords(keywords, indexes):

	mesh_counts = {}
	for keyword in keywords:
		if keyword in indexes['sorted_ent_cls_index']:
			for mesh_count in indexes['sorted_ent_cls_index'][keyword].items():
				mesh = mesh_count[0]
				count = mesh_count[1]
				if mesh in mesh_counts:
					mesh_counts[mesh] += count
				else:
					mesh_counts[mesh] = count
	# Sort the meshes by count
	mesh_counts = dict(sorted(mesh_counts.items(), key=lambda item: item[1], reverse=True))

	return mesh_counts


def get_mesh_counts_from_meshes(meshes):

	mesh_counts = {}
	for mesh_item in meshes:
		try:
			for mesh in mesh_item['ete']:
				if mesh in mesh_counts:
					mesh_counts[mesh] += 1
				else:
					mesh_counts[mesh] = 1
		except Exception as e:
			pass
	# Sort the meshes by count
	mesh_counts = dict(sorted(mesh_counts.items(), key=lambda item: item[1], reverse=True))

	return mesh_counts


def predict_keyword_meshes(dataset, indexes):

	predictions = {}

	#for record in dataset:
	json_lines_index = 0
	json_lines_total = 50070 # test (zbmed) # 4374874 (zbmath) # 17900000 (approx zbmed)
	with jsonlines.open(json_lines_path, 'r') as f:
		#print('Counting the number of lines...')
		#json_lines_total = sum(1 for _ in f)
		#print('Total lines: ' + str(json_lines_total))
		for record in f:
			# json_objects.index(record_line)
			if json_lines_index % 100 == 0:
				# # Print the progress percentage
				# if dataset.index(record) % 100 == 0:
				# 	print('Progress: ' + str(round(dataset.index(record) / len(dataset) * 100, 2)) + '%')
				# Display progress
				print('Progress: ' + str(round(json_lines_index / json_lines_total * 100, 2)) + '%')
			if json_lines_index % 100000 == 0:
				# Save intermediate results
				print('Intermediate saving at index: ' + str(json_lines_index))
				predictions_df = pd.DataFrame(predictions)
				predictions_df = predictions_df.transpose()
				predictions_df.to_csv(f'{zbmed_path}/meshes_prediction_table_cutoff{nr_meshes_cutoff}_tmp.csv', sep=';',
									  index=False)

			# Get the record ID
			dbrecordid = record['dbrecordid']

			# Get the keywords from the abstract
			abstract = record['abstract']
			keywords_text = get_keywords(abstract, indexes)

			# Get the actual keywords and meshes
			keywords_actual = [mesh['mh'] for mesh in record['mesh'] if mesh['mh'] is not None]
			meshes_actual = [mesh['enr'] for mesh in record['mesh'] if mesh['enr'] is not None]
			#meshes_actual = get_mesh_counts_from_meshes(meshes_actual)
			#meshes_actual = dict(sorted(meshes_actual.items(), key=lambda item: item[1], reverse=True))
			# meshes_actual_cutoff = list(meshes_actual)[:nr_meshes_cutoff]
			# meshes actual are unranked!

			# Predict the classes from the text keywords
			meshes_predicted_text = get_mesh_counts_from_keywords(keywords_text, indexes)

			# Predict the classes from the actual keywords
			meshes_predicted_keywords = get_mesh_counts_from_keywords(keywords_actual, indexes)

			# Apply the cutoff
			meshes_predicted_text_cutoff = list(meshes_predicted_text)[:nr_meshes_cutoff]
			meshes_predicted_keywords_cutoff = list(meshes_predicted_keywords)[:nr_meshes_cutoff]

			# Get the overlap between the actual and predicted meshes
			# meshes text vs. actual
			overlap_meshes_text_and_actual = [mesh for mesh in meshes_predicted_text_cutoff if mesh in meshes_actual]
			if len(meshes_actual) > 0:
				overlap_count_meshes_text_and_actual = len(overlap_meshes_text_and_actual) / len(meshes_actual)
			else:
				overlap_count_meshes_text_and_actual = 0
			overlap_meshes_keywords_and_actual = [mesh for mesh in meshes_predicted_keywords_cutoff if mesh in meshes_actual]
			# meshes keywords vs. actual
			if len(meshes_actual) > 0:
				overlap_count_meshes_keywords_and_actual = len(overlap_meshes_keywords_and_actual) / len(meshes_actual)
			else:
				overlap_count_meshes_keywords_and_actual = 0

			# Store the predictions
			predictions[dbrecordid] = {
				'id': dbrecordid,
				'keywords_extracted_text': keywords_text,
				'keywords_actual': keywords_actual,
				'predicted_from_text': meshes_predicted_text_cutoff,
				'predicted_from_keywords': meshes_predicted_keywords_cutoff,
				'meshes_actual': meshes_actual, #_cutoff
				'overlap_meshes_text_and_actual': overlap_count_meshes_text_and_actual,
				'overlap_meshes_keywords_and_actual': overlap_count_meshes_keywords_and_actual
			}

			json_lines_index += 1

			if json_lines_index > json_lines_total:
				break

	# Save the predictions to a CSV file
	predictions_df = pd.DataFrame(predictions)
	predictions_df = predictions_df.transpose()
	predictions_df.to_csv(f'{zbmed_path}/meshes_prediction_table_cutoff{nr_meshes_cutoff}.csv', sep=';', index=False)

	# Print mean overlaps
	print('Mean overlap meshes predicted from text vs. actual: ' + str(np.mean([predictions[record]['overlap_meshes_text_and_actual'] for record in predictions])))
	print('Mean overlap meshes predicted from keywords vs. actual: ' + str(np.mean([predictions[record]['overlap_meshes_keywords_and_actual'] for record in predictions])))

	return predictions

def get_precision_recall_curves_meshes():

	# load prediction table
	prediction_table = pd.read_csv(f'{zbmed_path}/meshes_prediction_table_cutoff{nr_meshes_cutoff}.csv', sep=';')

	# evaluate predictions
	# ir measures
	# competing origins = text vs. keywords
	meshes_origins = ['predicted_from_text', 'predicted_from_keywords']
	precision_recall = {meshes_origins[0]: {}, meshes_origins[1]: {}}

	latest_progress = 0
	for idx in range(len(prediction_table)):

		current_progress = round(idx / len(prediction_table) * 100, 1)
		if current_progress != latest_progress and current_progress % 10 == 0:
			print(current_progress, '%')
			latest_progress = current_progress

		# collect meshes
		meshes_dict = {}

		# document nr.
		# id = prediction_table['id'][idx]

		# meshes (actual) = baseline
		meshes_actual = prediction_table['meshes_actual'][idx]#[:nr_meshes_cutoff]
		meshes_dict['meshes_actual'] = meshes_actual.replace("'", "").lstrip('[').rstrip(']').strip().split(', ')

		# meshes (predicted_text) = competitor
		meshes_predicted_text = prediction_table['predicted_from_text'][idx]
		meshes_dict['predicted_from_text'] = meshes_predicted_text.replace("'", "").lstrip('[').rstrip(']').strip().split(
			', ')

		# meshes (predicted_keywords) = competitor
		meshes_predicted_keywords = prediction_table['predicted_from_keywords'][idx]
		meshes_dict['predicted_from_keywords'] = meshes_predicted_keywords.replace("'", "").lstrip('[').rstrip(
			']').strip().split(', ')

		# iterate over meshes
		try:

			# meshes (actual) = baseline
			meshes_actual = meshes_dict['meshes_actual']

			for meshes_origin in meshes_origins:

				meshes_predicted_full = meshes_dict[meshes_origin]

				for i in range(1,nr_meshes_cutoff + 1):
					meshes_predicted = meshes_predicted_full[:i]
					# https://stats.stackexchange.com/questions/21551/how-to-compute-precision-recall-for-multiclass-multilabel-classification
					# precision = ratio of how much of the predicted is correct
					meshes_intersection = [mesh for mesh in meshes_predicted if mesh in meshes_actual]
					if len(meshes_predicted) > 0:
						precision = len(meshes_intersection) / len(meshes_predicted)
					else:
						precision = 1
					# recall = ratio of how many of the actual labels were predicted
					if len(meshes_actual) > 0:
						recall = len(meshes_intersection) / min(len(meshes_actual),nr_meshes_cutoff)
					else:
						recall = 1

					# store metrics
					try:
						precision_recall[meshes_origin][i]['precisions'].append(precision)
						precision_recall[meshes_origin][i]['recalls'].append(recall)
					except:
						try:
							precision_recall[meshes_origin][i]['precisions'] = [precision]
							precision_recall[meshes_origin][i]['recalls'] = [recall]
						except:
							precision_recall[meshes_origin][i] = {'precisions': [precision], 'recalls': [recall]}
		except:
			print('Error at index:', idx)

	# plot metrics
	# precision-recall curve
	fig, ax = plt.subplots()
	# collect metrics for plot
	for meshes_origin in meshes_origins:

		pr_rc = precision_recall[meshes_origin]
		precisions = []
		recalls = []
		cutoffs = []
		for p_r in pr_rc.items():
			cutoffs.append(p_r[0])
			precisions.append(np.mean(p_r[1]['precisions']))
			recalls.append(np.mean(p_r[1]['recalls']))

		marker_dict = {meshes_origins[0]: 'x',
					   meshes_origins[1]: 's'}  # , meshes_origins[2]: 'o', meshes_origins[3]: '>'}
		ax.scatter(recalls, precisions, marker=marker_dict[meshes_origin], s=10, label=meshes_origin)

	# Center the legend
	legend = ax.legend(loc='center', bbox_to_anchor=(0.5, 0.5), fontsize=12)
	# Ensure the legend box has a background and frame
	legend.get_frame().set_facecolor('white')
	legend.get_frame().set_edgecolor('black')

	# # set plot parameters
	# plt.xlim(0, 0.8)
	# plt.ylim(0, 0.8)

	# set plot labels

	plt.title('Precision-Recall Curve for MeSH Unique ID Prediction')
	plt.xlabel('Recall')
	plt.ylabel('Precision')
	plt.savefig(f'{zbmed_path}/prec-rec-curve_meshes.pdf', format="pdf", bbox_inches="tight")
	plt.show()

###########
# EXECUTE #
###########

# Set parameters
nr_meshes_cutoff = 10
only_plot = True # False # True

# Execute pipeline
if not only_plot:
	#dataset = load_dataset()
	indexes = load_indexes()
	_ = predict_keyword_meshes(None,indexes)
# plot metrics
get_precision_recall_curves_meshes()
