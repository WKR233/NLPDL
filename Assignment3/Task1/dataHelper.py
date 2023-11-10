import datasets
import transformers
import json
import itertools

S=' '
class Dataset:
	def __init__(self, textlist, labellist):
		self.Dict['text'] = textlist
		self.Dict['labels'] = labellist

class DatasetDict:
	def __init__(self, train, test):
		self.Dict['train'] = train
		self.Dict['test'] = test

def get_text(Dict):
	return Dict['term']+' '+S+Dict['sentence']

def get_label(Dict):
	return Dict['polarity']

def get_subdataset(subdataset_name):
	if(subdataset_name=="restaurant" or subdataset_name=="laptop"):
		if(subdataset_name=="restaurant"):
			train_file_path="./SemEval14-res/train.json"
			test_file_path="./SemEval14-res/test.json"
			
		elif(subdataset_name=="laptop"):
			train_file_path="./SemEval14-laptop/train.json"
			test_file_path="./SemEval14-laptop/test.json"

		with open(train_file_path, 'r') as train_file:
			with open(test_file_path, 'r') as test_file:
				train=json.load(train_file)
				test=json.load(test_file)
				
				trainlabellist=list(map(get_label, (list)(train.values())))
				testlabellist=list(map(get_label, (list)(test.values())))

				traintextlist=list(map(get_text, (list)(train.values())))
				testtextlist=list(map(get_text, (list)(test.values())))

		train_dataset=Dataset(traintextlist, trainlabellist)
		test_dataset=Dataset(testtextlist, testlabellist)
		subdataset=DatasetDict(train_dataset, test_dataset)
	
	elif(subdataset_name=="acl"):
		train_file_path="./acl-arc/train.jsonl"
		test_file_path="./acl-arc/test.jsonl"

		traintextlist=[]
		trainlabellist=[]
		testtextlist=[]
		testlabellist=[]

		with open(train_file_path, 'r') as train_file:
			with open(test_file_path, 'r') as test_file:
				#line here is a json file, and data is a dict
				for train_line in train_file:
					traintextlist.append(json.loads(train_line)['text'])
					trainlabellist.append(json.loads(train_line)['intent'])
				for test_line in test_file:
					testtextlist.append(json.loads(test_line)['text'])
					testlabellist.append(json.loads(test_line)['intent'])

	elif(subdataset_name=="agnews"):
		dataset=datasets.load_dataset("agnews")
		train_dataset=dataset.train_test_split(test_size=0.1, train_size=0.9, seed=2022)
	return subdataset

def get_dataset(dataset_name, sep_token):
	'''
	dataset_name: str, the name of the dataset
	sep_token: str, the sep_token used by tokenizer(e.g. '<sep>')
	'''
	S=sep_token
	if(type(dataset) != list):
		if(dataset_name.endswith("sup")):
			substr = dataset_name[:-4]

		elif(dataset_name.endswith("fs")):
			substr = dataset_name[:-3]
	
	else:
		print(0)

	dataset = 0

	return dataset