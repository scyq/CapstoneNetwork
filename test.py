import re
import jieba

re_pattern = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")

text = "我真的123Aza"

text = re_pattern.sub("", text)

text = jieba.lcut(text)

print(text)
print(len(text))