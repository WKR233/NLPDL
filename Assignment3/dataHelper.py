import datasets
import transformers
import json
import itertools
import pandas as pd
from datasets import Dataset, DatasetDict

def get_subdataset(subdataset_name, sep_token, label_offset=0):
	
	traintextlist=[]
	trainlabellist=[]
	testtextlist=[]
	testlabellist=[]

	if 'restaurant' in subdataset_name or 'laptop' in subdataset_name:
		if('restaurant' in subdataset_name):
			train_file_path="./SemEval14-res/train.json"
			test_file_path="./SemEval14-res/test.json"
			
		elif('laptop' in subdataset_name):
			train_file_path="./SemEval14-laptop/train.json"
			test_file_path="./SemEval14-laptop/test.json"

		with open(train_file_path, 'r') as train_file:
			with open(test_file_path, 'r') as test_file:
				train=json.load(train_file)
				test=json.load(test_file)
				
				for value in (list)(train.values()):
					if value['polarity']=='positive':
						trainlabellist.append(0+label_offset)
					elif value['polarity']=='neutral':
						trainlabellist.append(1+label_offset)
					else:
						trainlabellist.append(2+label_offset)
					traintextlist.append(value['term']+' '+sep_token+value['sentence'])

				for value in (list)(test.values()):
					if value['polarity']=='positive':
						testlabellist.append(0+label_offset)
					elif value['polarity']=='neutral':
						testlabellist.append(1+label_offset)
					else:
						testlabellist.append(2+label_offset)
					testtextlist.append(value['term']+' '+sep_token+value['sentence'])

		train_dataset=Dataset.from_dict({'text': traintextlist, 'labels': trainlabellist})
		test_dataset=Dataset.from_dict({'text': testtextlist, 'labels': testlabellist})
		subdataset=DatasetDict(
			{
				'train': train_dataset, 
				'test': test_dataset
			}
		)
	
	elif 'acl' in subdataset_name:
		train_file_path="./acl-arc/train.jsonl"
		test_file_path="./acl-arc/test.jsonl"
		with open(train_file_path, 'r') as train_file:
			with open(test_file_path, 'r') as test_file:
				for train_line in train_file:
					traintextlist.append(json.loads(train_line)['text'])
					trainlabellist.append(json.loads(train_line)['intent']+label_offset)
				for test_line in test_file:
					testtextlist.append(json.loads(test_line)['text'])
					testlabellist.append(json.loads(test_line)['intent']+label_offset)

		train_dataset=Dataset.from_dict({'text': traintextlist, 'labels': trainlabellist})
		test_dataset=Dataset.from_dict({'text': testtextlist, 'labels': testlabellist})
		subdataset=DatasetDict(
			{
				'train': train_dataset,
				'test': test_dataset
			}
		)

	elif 'agnews' in subdataset_name:
		Dict={}
		textlist=[]
		labellist=[]
		with open("./agnews/test.jsonl", 'r') as file:
			for line in file:
				textlist.append(json.loads(line)['text'])
				labellist.append(json.loads(line)['label']+label_offset)

		subdataset=Dataset.from_dict({'text': textlist, 'labels': labellist})
		subdataset=subdataset.train_test_split(test_size=0.1, seed=2022, shuffle=True)
	
	if 'fs' in subdataset_name:
		seed = 2022
		num_labels = max(subdataset['train']['labels']) - label_offset

		if num_labels < 4:
			subdataset['train'] = subdataset['train'].shuffle(seed=seed)
			subdataset['train'] = subdataset['train'].select(range(32))
		else:
			subdataset['train'] = subdataset['train'].shuffle(seed=seed)
			_idx = [[] for i in range(num_labels+1)]
			for idx, label in enumerate(subdataset['train']['labels']):
				if len(_idx[label]) < 8:
					_idx[label].append(idx)
			idx_lst = [i for item in _idx for i in item]
			subdataset['train'] = subdataset['train'].select(idx_lst).shuffle(seed=seed)

	return subdataset

def get_dataset(dataset_name, sep_token):
	'''
	dataset_name: str, the name of the dataset
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
	'''
	if type(dataset_name)!=list:
		print("1")
		return get_subdataset(dataset_name, sep_token)
	else:
		label_offset=0
		train_dataset_list=[]
		test_dataset_list=[]
		for dataset in dataset_name:
			subdataset=get_subdataset(dataset, sep_token, label_offset)
			label_offset=max(subdataset['train']['labels'])+1
			train_dataset_list.append(subdataset['train'])
			test_dataset_list.append(subdataset['test'])
		return DatasetDict({'train': datasets.concatenate_datasets(train_dataset_list),
					  		'test': datasets.concatenate_datasets(test_dataset_list)})
	
def main():
	test1=get_dataset(["agnews_fs", "restaurant_fs"], '<SEP>')
	train=test1['train']
	text=train['text']
	print(text[0])

if __name__ == '__main__':
	main()