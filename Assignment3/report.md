<font size=10>Assignment3 Report</font>

<font size=5>2100017727 王柯然</font>

# Task1 Dataset Building
参考dataHelper.py文件

# Task2 Training Script
参考train.py与train.bash

我们为了准确性，每一组数据-模型都用5个不同的seed进行实验，具体的数据位于task2.xlsx，而数据的整体特征如下：

## loss

| MEAN | roberta-base | bert-base-uncased | allenai/scibert_scivocab_uncased |
|:-:|:-:|:-:|:-:|
| restaurant_sup | 0.8700 | 0.9755 | 1.118 |
| acl_sup | 1.030 | 1.121 | 1.271 |
| agnews_sup | 0.4775 | 0.4834 | 0.5360 |

| VARIANCE | roberta-base | bert-base-uncased | allenai/scibert_scivocab_uncased |
|:-:|:-:|:-:|:-:|
| restaurant_sup | 0.1288 | 0.1599 | 0.2102 |
| acl_sup | 0.1836 | 0.2236 | 0.2768 |
| agnews_sup | 0.03870 | 0.03842 | 0.04778 |

## accuracy

| MEAN | roberta-base | bert-base-uncased | allenai/scibert_scivocab_uncased |
|:-:|:-:|:-:|:-:|
| restaurant_sup | 0.8648 | 0.8372 | 0.8270 |
| acl_sup | 0.8119 | 0.7712 | 0.7610 |
| agnews_sup | 0.9327 | 0.9283 | 0.9214 |

| VARIANCE | roberta-base | bert-base-uncased | allenai/scibert_scivocab_uncased |
|:-:|:-:|:-:|:-:|
| restaurant_sup | 0.1247 | 0.1397 | 0.1142 |
| acl_sup | 0.1105 | 0.09879 | 0.09622 |
| agnews_sup | 0.1453 | 0.1433 | 0.1416 |

## micro_f1

| MEAN | roberta-base | bert-base-uncased | allenai/scibert_scivocab_uncased |
|:-:|:-:|:-:|:-:|
| restaurant_sup | 0.8648 | 0.8372 | 0.8270 |
| acl_sup | 0.8119 | 0.7712 | 0.7610 |
| agnews_sup | 0.9327 | 0.9283 | 0.9214 |

| VARIANCE | roberta-base | bert-base-uncased | allenai/scibert_scivocab_uncased |
|:-:|:-:|:-:|:-:|
| restaurant_sup | 0.1247 | 0.1397 | 0.1142 |
| acl_sup | 0.1105 | 0.09879 | 0.09622 |
| agnews_sup | 0.1453 | 0.1433 | 0.1416 |

## macro_f1

| MEAN | roberta-base | bert-base-uncased | allenai/scibert_scivocab_uncased |
|:-:|:-:|:-:|:-:|
| restaurant_sup | 0.7908 | 0.7498 | 0.7334 |
| acl_sup | 0.7742 | 0.7160 | 0.7079 |
| agnews_sup | 0.9314 | 0.9266 | 0.9198 |

| VARIANCE | roberta-base | bert-base-uncased | allenai/scibert_scivocab_uncased |
|:-:|:-:|:-:|:-:|
| restaurant_sup | 0.1041 | 0.09266 | 0.09025 |
| acl_sup | 0.1007 | 0.08489 | 0.08319 |
| agnews_sup | 0.1449 | 0.1432 | 0.1411 |

训练的图像如下所示（为了简化只列出了其中一种情况）：

![accuracy](graph/model/accuracy.png)
![macro_f1](graph/model/macro_f1.png)
![loss](graph/model/loss.png)

## 结论
1. 对于不同的模型，数据集agnews拥有最好的表现
2. 对于不同的数据集，模型roberta-base拥有最好的表现
3. accuracy永远与micro_f1相等

# Task3 Parameter-efficient Fine-tuning(PEFT)

参考./adapter中的adapter.py文件，以及lora.py文件，具体实验结果如下：

## Adapter

| MEAN | loss | accuracy | micro_f1 | macro_f1 |
|:-:|:-:|:-:|:-:|:-:|
| restaurant_sup | 0.4674 | 0.8616 | 0.8616 | 0.7798 |
| acl_sup | 0.5125 | 0.8041 | 0.8041 | 0.7654 |
| agnews_sup | 0.253 | 0.9250 | 0.9250 | 0.9232 |

## LoRA

| MEAN | loss | accuracy | micro_f1 | macro_f1 |
|:-:|:-:|:-:|:-:|:-:|
| restaurant_sup | 0.4344 | 0.8196 | 0.8196 | 0.6883 |
| acl_sup | 1.021 | 0.6259 | 0.6259 | 0.2797 |
| agnews_sup | 0.2283 | 0.9197 | 0.9197 | 0.9179 |

训练的图像如下所示：

## Adapter Graphs

![agnews-accuracy](graph/adapter/acl-accuracy.png)
![agnews-loss](graph/adapter/acl-loss.png)
![agnews-macro_f1](graph/adapter/acl-macro_f1.png)
![acl-accuracy](graph/adapter/acl-accuracy.png)
![acl-loss](graph/adapter/acl-loss.png)
![acl-macro_f1](graph/adapter/acl-macro_f1.png)
![rest-accuracy](graph/adapter/rest-accuracy.png)
![rest-loss](graph/adapter/rest-loss.png)
![rest-macro_f1](graph/adapter/rest-macro_f1.png)

## LoRA Graphs

![agnews-accuracy](graph/lora/agnews-accuracy.png)
![agnews-macro_f1](graph/lora/agnews-macro_f1.png)
![agnews-loss](graph/lora/agnews-loss.png)
![acl-accuracy](graph/lora/acl-accuracy.png)
![acl-macro_f1](graph/lora/acl-macro_f1.png)
![acl-loss](graph/lora/acl-loss.png)
![rest-accuracy](graph/lora/rest-accuracy.png)
![rest-macro_f1](graph/lora/rest-macro_f1.png)
![rest-macro_f1](graph/lora/rest-loss.png)

# 其他计算

+ If you directly fine-tune LLaMA-3b without PEFT, how much GPU memory do you need? (Please estimate it, and do not run the experiment)

$$
4n+(34bh+5b^2a)l
$$
我们取
$$
h=3200, a=32, l=26, b=64, n=3\times 10^9
$$
即需要约45.44GB的空间

+ With PEFT, how much GPU memory is saved?

LoRA的可训练参数大约有30M参数，我们需要存储原始模型、梯度、中间激活层等参数。同时假定LoRA的中间激活层与原始模型相同，则大约有24.27GB需要存储，节省了21.17GB