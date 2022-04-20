import json
import re
import csv

clean_chat_data = "clean_chat_corpus/qa_data.tsv"
delimiter = "\t"


# 下函数仅针对PysQA数据集
def json2tsv():
    '''
        我猜测数据集是从一些论坛的问答数据中抽取的
        因此部分answer中包含了多层QA
        一条answer_text好像有多层楼的回答
        被很多特殊符号分隔开了
        主要是中英文井号
        所以利用正则表达式只取第一层回复
    '''
    IF_SPLIT_DATA = True  # 如果要去除多层回复，把IF_SPLIT_DATA改为True
    IF_DATA_AUGMENTATION = False  # 如果要增加训练数据，把问题描述也算进去，把IF_DATA_AUGMENTATION改为True
    IF_UNIQUE_KEY = True  # 如果要去除重复的问题，把IF_UNIQUE_KEY改为True，改为False可能会不说人话
    re_splitter = "#|＃|■|※|@|<|>"

    # 该数据集涉密, 所以不能直接提取
    with open("PsyQA_full.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)

    with open(clean_chat_data, 'w', encoding='utf-8', newline="") as f:
        tsv_writer = csv.writer(f, delimiter=delimiter)
        for obj in json_data:
            questions = []
            # 记得去除delimiter
            questions.append(obj["question"].replace(delimiter, ""))  # 简洁的问题
            if IF_DATA_AUGMENTATION:
                questions.append(obj["description"].replace(delimiter, ""))
            answers = obj["answers"]
            for question in questions:
                for answer in answers:
                    answer_text = answer["answer_text"].replace(delimiter, "")
                    if (IF_SPLIT_DATA and re.search(re_splitter, answer_text)):
                        answer_text = re.split(re_splitter, answer_text)
                        # 去除列表中的空字符串
                        answer_text = [
                            x.strip() for x in answer_text if len(x) > 0
                        ][0]
                    to_write = [question, answer_text]
                    tsv_writer.writerow(to_write)
                    if IF_UNIQUE_KEY:  # 只执行一遍
                        break

    print("json转tsv完成")


if __name__ == "__main__":
    json2tsv()