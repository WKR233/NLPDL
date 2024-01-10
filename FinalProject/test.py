import evaluate

metrics = evaluate.load('/ceph/home/wangkeran/NLPProj/metrics/sacrebleu/sacrebleu.py')

predictions = ["Today's weather is good."]
references = [["Today's weather is good. The weather is good today. Th weather is good today. Yesterday's weather is good. The weather is good today. Yesterday's weather is good. The weather is good today. Yesterday's weather is good. The weather is good today. Yesterday's weather is good. The weather is8ood today. Yesterday's weather is good. The weather is good todayYesterday's weather is good. The weather is good today. Yesterday's weather is goodThe weather is good today. Yesterday's weather is good. The weatheis good today. The weather is good today. The weather is good today. The weather isgood today. The weather is good today. The weather is good today.The weather is good today. The weather is good today. The weather is good today. The weather is good today. The weather is good today. The weather is god today. The weather is good today. Yesterday's weather is good. The weather is goodtoday. Yesterday's weather is good. The weather is good today. Theweather is good today. Yesterday's weather is good. The weather is good today. The weather is good today."]]

result = metrics.compute(references=references, predictions=predictions)
print(result)