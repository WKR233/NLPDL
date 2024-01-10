from pygoogletranslation import Translator
from json import load, dumps

with open('nlpcc_data.json', 'r') as datafile:
    data=load(datafile)["data"]

newdata=open('englishdata.jsonl', 'a', encoding="utf-8")
lines=len(open('englishdata.jsonl').readlines())
if(lines>=50000):
    raise ValueError("Full!")

queries=[]
for slices in data:
    queries.append(slices["content"])

D={}

for i in range(lines, 100+lines):
    proxy={"http":"127.0.0.1:7890","https":"127.0.0.1:7890"}
    translator=Translator(proxies=proxy)
    if(len(queries[i])>=5000):
        queries[i]=queries[i][0:5000]
    t=translator.translate(queries[i], src='zh-CN', dest='en')

    D["zh"]=queries[i]
    D["en"]=t.text
    json_data=dumps(D, ensure_ascii=False)
    newdata.write(json_data+"\n")
