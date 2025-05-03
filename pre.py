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
    model = Word2Vec(sentences, vector_size=cfg.word2vec_emb_dim, window=5, min_count=1, workers=4)
    # 将每个文本转换为向量
    word2vec_matrix = np.array([
        np.mean([model.wv[word] for word in words if word in model.wv] or [np.zeros(cfg.word2vec_emb_dim)], axis=0)
        for words in sentences
    ])
    print_green("Word2Vec 向量化完成。")
    return word2vec_matrix, model

@pkl_cache(cfg.logdir + f"/{cfg.emb_method}_vectorizer.pkl")
def vectorize_text_seqword2vec(content):
    print_blue("正在进行文本向量化 (seqWord2Vec)...")
    if cfg.cls_method != "lstm":
        raise ValueError(f"不支持的分类方法: {cfg.cls_method}")
    
    sentences = [text.split() for text in content]
    
    
    # if False:
    #     from gensim.scripts.glove2word2vec import glove2word2vec

    #     glove2word2vec('glove.6B.300d.txt', 'glove.6B.300d.word2vec.txt')
    #     model = KeyedVectors.load_word2vec_format('glove.6B.300d.word2vec.txt', binary=False)
    if cfg.word2vec_emb_dim == 300:
        from gensim.models import KeyedVectors
        model = KeyedVectors.load_word2vec_format('./word2vec/GoogleNews-vectors-negative300.bin', binary=True)
        maxlen = 100
    elif cfg.word2vec_emb_dim == 50:
        from gensim.models import KeyedVectors
        model = KeyedVectors.load_word2vec_format("./word2vec/glove_50d_w2v_format.txt", binary=False)
        maxlen = 75
        # maxlen = 150  # X
        # maxlen = 500  # X 训练不稳定，而且很难收敛
    else:
        raise ValueError(f"不支持的词向量维度: {cfg.word2vec_emb_dim}")

    
    # 构建词汇表到索引的映射
    # word_to_idx = {word: idx+1 for idx, word in enumerate(model.wv.index_to_key)}  # 0保留给padding
    
    # 将文本转换为单词索引序列
    word_indices = []
    for words in tqdm(sentences):
        seq = []
        for word in words:
            if word in model:
                seq.append((model.get_vector(word, norm=True)).astype(np.float32))
            else:
                seq.append(np.zeros((cfg.word2vec_emb_dim,), dtype=np.float32))
            assert isinstance(seq[-1], np.ndarray)
            assert len(seq[-1]) == cfg.word2vec_emb_dim
        word_indices.append(seq)
    from lstm import pad_sequences
    word_indices = pad_sequences(word_indices, maxlen=maxlen, value=np.zeros(cfg.word2vec_emb_dim,))
    
    print_green("Word2Vec 向量化完成。")
    # return {'word_indices':word_indices, 'word_to_idx': word_to_idx,}, model
    return word_indices, model
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
    elif cfg.emb_method == "seqword2vec":
        return vectorize_text_seqword2vec(content)
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
    elif cfg.emb_method == "seqword2vec":
        # word_indices, word_to_idx = content['word_indices'], content['word_to_idx']
        # X_train_, X_test_, y_train, y_test = train_test_split(
        #     word_indices, label, test_size=test_size, random_state=cfg.rs
        # )
        # X_train = {'word_indices':X_train_, 'word_to_idx': word_to_idx}
        # X_test = {'word_indices':X_test_, 'word_to_idx': word_to_idx}

        X_train, X_test, y_train, y_test = train_test_split(
            content, label, test_size=test_size, random_state=cfg.rs
        )
    else:
        # 其他向量化方法
        X_train, X_test, y_train, y_test = train_test_split(
            content, label, test_size=test_size, random_state=cfg.rs
        )
    
    print_green("训练集和测试集划分完成。")
    return X_train, X_test, y_train, y_test

