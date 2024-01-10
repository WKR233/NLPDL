from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoConfig
from peft import PeftModel, PeftConfig
import evaluate
metrics = evaluate.load("./metrics/sacrebleu")
peft_model_id = "./mT5-englishdata/checkpoint-9000"
config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
model = PeftModel.from_pretrained(model=model, model_id=peft_model_id)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
reference = tokenizer.encode("Today's weather is good")

def Model(text):
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output = model.generate(inputs=input_ids, max_length=500)
    result = metrics.compute(predictions=output[0], references=reference)
    return tokenizer.decode(output[0], skip_special_tokens=True)

ans = Model("今天的天气很好")
print(ans)