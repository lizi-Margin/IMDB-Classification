"""
    pre.py
    by shc 2025.5.3
"""
import re
import numpy as np
from collections import Counter
# import nltk
# from nltk.corpus import stopwords
from tqdm import tqdm
from UTIL.colorful import print_blue, print_green, print_yellow
from global_config import GlobalConfig as cfg
from pkl_cache import pkl_cache

# nltk.download('stopwords')
# STOPWORDS_PATH = nltk.data.find('corpora/stopwords') + "/english"
STOPWORDS_PATH = '/home/hulc/nltk_data/' + 'corpora/stopwords' + "/english"

# (一) 构建词汇表
def build_vocabulary(content, min_freq=5):
    print_blue("正在构建词汇表...")
    word_count = Counter()
    for text in tqdm(content, desc="统计词频"):
        words = text.split()
        word_count.update(words)
    
    # 过滤低频词并按频率排序
    filtered_words = [word for word, freq in word_count.items() if freq >= min_freq]
    sorted_words = sorted(filtered_words, key=lambda x: word_count[x], reverse=True)
    
    # 构建词汇表
    vocab = {word: idx for idx, word in enumerate(sorted_words)}
    print_green("词汇表构建完成。")
    return vocab

# (三) 移除特殊字符
def remove_special_characters(text):
    clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    return clean_text

# (四) 分词
def tokenize(text):
    words = text.lower().split()
    return words

# (五) 去除停用词
def remove_stopwords(words):
    with open(STOPWORDS_PATH, 'r') as f:
        stop_words = set(f.read().splitlines())
    # print(stop_words)
    filtered_words = [word for word in words if word not in stop_words]
    return filtered_words

# (六) 标签映射
def map_labels(label):
    return 1 if label == 'positive' else 0

# (七) 文本向量化（选择TF-IDF）
@pkl_cache(cfg.logdir + f"/{cfg.emb_method}_vectorizer.pkl")
def vectorize_text_bow(content):
    print_blue("正在进行文本向量化 (BOW)...")
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    bow_matrix = vectorizer.fit_transform(content)
    print_green("BOW 向量化完成。")
    return bow_matrix, vectorizer

@pkl_cache(cfg.logdir + f"/{cfg.emb_method}_vectorizer.pkl")
def vectorize_text_tfidf(content):
    print_blue("正在进行文本向量化 (TF-IDF)...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(content)
    print_green("TF-IDF 向量化完成。")
    return tfidf_matrix, vectorizer

@pkl_cache(cfg.logdir + f"/{cfg.emb_method}_vectorizer.pkl")
def vectorize_text_word2vec(content):
    print_blue("正在进行文本向量化 (Word2Vec)...")
    from gensim.models import Word2Vec
    # 将文本转换为单词列表
    sentences = [text.split() for text in content]
    # 训练 Word2Vec 模型
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    # 将每个文本转换为向量
    word2vec_matrix = np.array([
        np.mean([model.wv[word] for word in words if word in model.wv] or [np.zeros(100)], axis=0)
        for words in sentences
    ])
    print_green("Word2Vec 向量化完成。")
    return word2vec_matrix, model

@pkl_cache(cfg.logdir + f"/{cfg.emb_method}_vectorizer.pkl")
def vectorize_text_bert(content):
    print_blue("正在进行文本向量化 (BERT)...")
    if cfg.cls_method != "bert":
        raise ValueError(f"不支持的分类方法: {cfg.cls_method}")
    from transformers import BertTokenizer, BertTokenizerFast, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("pretrained/bert-base-uncased")
    # 换为BERT输入格式
    encoded_inputs = tokenizer(content, padding=True, truncation=True, return_tensors="pt")
    print_green("BERT 向量化完成。")
    # print(encoded_inputs)
    return encoded_inputs, tokenizer

def vectorize_text(content):
    from global_config import GlobalConfig as cfg
    if cfg.emb_method == "bow":
        return vectorize_text_bow(content)
    elif cfg.emb_method == "tfidf":
        return vectorize_text_tfidf(content)
    elif cfg.emb_method == "word2vec":
        return vectorize_text_word2vec(content)
    elif cfg.emb_method == "bert":
        return vectorize_text_bert(content)
    else:
        raise ValueError(f"不支持的嵌入方法: {cfg.emb_method}")

# (八) 划分训练集和测试集
def split_train_test(content, label, test_size):
    print_blue("正在划分训练集和测试集...")
    from sklearn.model_selection import train_test_split
    
    # 难绷的补丁
    if cfg.emb_method == "bert":
        # length = len(content['input_ids'])
        # y_train = label[:int(length*(1-test_size))]
        # y_test = label[int(length*(1-test_size)):]
        # X_train, X_test = {}, {}
        # for key in content:
        #     X_train[key], X_test[key] = content[key][:int(length*(1-test_size))], content[key][int(length*(1-test_size)):]
        X_train = content
        X_test = content
        y_train = label
        y_test = label
    else:
        # 其他向量化方法
        X_train, X_test, y_train, y_test = train_test_split(
            content, label, test_size=test_size, random_state=cfg.rs
        )
    
    print_green("训练集和测试集划分完成。")
    return X_train, X_test, y_train, y_test

