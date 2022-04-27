# from regex import P
# import torch
# from transformers import BertTokenizer, BertConfig
# from transformers import BertForQuestionAnswering

# model_name = 'bert-base-chinese'

# # 通过词典导入分词器
# tokenizer = BertTokenizer.from_pretrained(model_name)
# # 导入配置文件
# model_config = BertConfig.from_pretrained(model_name)
# # 最终有两个输出，初始位置和结束位置
# model_config.num_labels = 2

# # 根据bert的model_config新建BertForQuestionAnswering
# model = BertForQuestionAnswering(model_config)
# model.eval()

# question, text = "里昂是谁", "里昂是一个杀手"

# sen_code = tokenizer.encode_plus(question, text)

# tokens_tensor = torch.tensor([sen_code['input_ids']])
# segments_tensor = torch.tensor([sen_code['token_type_ids']])

# start_pos, end_pos = model(tokens_tensor, segments_tensor)
# print(type(start_pos))
# print(end_pos)
# # 进行逆编码，得到原始的token
# all_tokens = tokenizer.convert_ids_to_tokens(sen_code['input_ids'])
# print(
#     all_tokens
# )  #['[CLS]', '里', '昂', '是', '谁', '[SEP]', '里', '昂', '是', '一', '个', '杀', '手', '[SEP]']

# # 对输出的答案进行解码的过程
# answer = ' '.join(all_tokens[torch.argmax(start_pos):torch.argmax(end_pos) +
#                              1])

# # 每次执行的结果不一致，这里因为没有经过微调，所以效果不是很好，输出结果不佳，下面的输出是其中的一种。
# print(answer)  #一 个 杀 手 [SEP]

from transformers import BertModel, BertTokenizer, BertConfig
import torch
from bert_model import ChatBot

model_name = "bert-base-chinese"

model_config = BertConfig.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)
model = ChatBot(model_config)

inputs = tokenizer("我好痛苦", return_tensors="pt")

question, text = "里昂是谁", "里昂是一个杀手"

sen_code = tokenizer.encode_plus(question, text)

tokens_tensor = torch.tensor([sen_code['input_ids']])
segments_tensor = torch.tensor([sen_code['token_type_ids']])

start_pos, end_pos = model(tokens_tensor, segments_tensor)
# 进行逆编码，得到原始的token
all_tokens = tokenizer.convert_ids_to_tokens(sen_code['input_ids'])
# print(
#     all_tokens
# )  #['[CLS]', '里', '昂', '是', '谁', '[SEP]', '里', '昂', '是', '一', '个', '杀', '手', '[SEP]']

# 对输出的答案进行解码的过程
answer = ' '.join(all_tokens[torch.argmax(start_pos):torch.argmax(end_pos) +
                             1])

# 每次执行的结果不一致，这里因为没有经过微调，所以效果不是很好，输出结果不佳，下面的输出是其中的一种。
print(answer)  #一 个 杀 手 [SEP]
