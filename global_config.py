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
    # emb_method = "bert"
    
    # 分类方法
    cls_method = "rf"
    # cls_method = "knn"
    # cls_method = "bert"
    cls_ldcpt = False
    
def print_config():
    print("GlobalConfig:")
    for key, value in vars(GlobalConfig).items():
        if not key.startswith("__"):
            print(f"  {key}: {value}")