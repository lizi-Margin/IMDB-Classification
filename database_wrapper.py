"""
    database_wrapper.py
    by shc 2025.5.3
"""
import re
from pre import *
from tqdm import tqdm
from UTIL.colorful import *
from collections import Counter


def print_list(lst, max_n=10):
    buf = f'list: value=['
    for i in range(len(lst)):
        if i >= max_n:
            buf += f'...'
            break
        buf += f'{repr(lst[i])}, '
    buf += f'], len={len(lst)} \n'
    print(buf, end='')


# (二) 设置路径及读取数据集
def read_IMDB():
    dataset_path="./database/IMDB/KaggleImdb/IMDB Dataset.csv"
    import pandas as pd
    print_blue("正在读取数据集...")
    df = pd.read_csv(dataset_path)
    content = df['review'].tolist()
    label = df['sentiment'].tolist()
    label = [1 if l == 'positive' else 0 for l in label]
    print_green("数据集读取完成。")
    return content, label

def get_IMDB():
    content, label = read_IMDB()
    
    # 数据预处理
    print_blue("正在进行数据预处理...")
    cleaned_content = []
    html_tags_counter = Counter() 
    for text in tqdm(content, desc="移除特殊字符"): 
        cleaned_text = remove_special_characters(text)
        cleaned_content.append(cleaned_text)

        # wtf = re.findall(r'[^a-zA-Z0-9\s]', cleaned_text)
        # html_tags_counter.update(wtf)
        tags_in_text = re.findall(r'<\/?[a-zA-Z][^>)]*[>)]', cleaned_text) 
        html_tags_counter.update(tags_in_text)
    for tag, count in html_tags_counter.most_common():  # 按出现次数排序
        print红(f"  {tag}: {count}次")
    

    tokenized_content = tokenize(cleaned_content)
    # tokenized_content = [tokenize(text) for text in tqdm(cleaned_content, desc="分词")]
    
    filtered_content = []
    for words in tqdm(tokenized_content, desc="去除停用词"):
        filtered_words = remove_stopwords(words)
        filtered_content.append(filtered_words)
    print_green("数据预处理完成。")

    # 词汇过滤
    filtered_content = vocabulary_filter(filtered_content)
    count_tokens(filtered_content)

    # list to str
    for i in range(len(filtered_content)):
        filtered_content[i] = " ".join(filtered_content[i])
    
    # 文本向量化
    vectorized_matrix, vectorizer = vectorize_text(filtered_content)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = split_train_test(vectorized_matrix, label, test_size=0.2)
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    X_train, X_test, y_train, y_test = get_IMDB()
    print_yellow("训练集特征示例：")
    print_list(X_train.toarray() if hasattr(X_train, 'toarray') else X_train)
    print_yellow("训练集标签示例：")
    print_list(y_train)