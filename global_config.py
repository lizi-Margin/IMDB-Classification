"""
    by shc 2025.5.3
"""
class GlobalConfig:
    device = "cuda:0"
    half = False
    NJ = 28
    rs = 42  # 随机

    logdir = "./result/"

    # 嵌入方法
    emb_method = "tfidf"
    # emb_method = "bow"
    # emb_method = "word2vec"
    # emb_method = "seqword2vec"
    # emb_method = "bert"
    # word2vec_emb_dim = 300  # 50GiB RAM Needed
    word2vec_emb_dim = 50
    
    # 分类方法
    # cls_method = "rf"
    cls_method = "knn"
    # cls_method = 'lstm'
    # cls_method = "bert"
    cls_ldcpt = False


    taskdir = f"{logdir}/{emb_method}-{cls_method}"
    
def print_config():
    print("GlobalConfig:")
    for key, value in vars(GlobalConfig).items():
        if not key.startswith("__"):
            print(f"  {key}: {value}")