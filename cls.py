import pickle
import numpy as np
from UTIL.colorful import print_blue, print_green, print_yellow
from global_config import GlobalConfig as cfg

def bert_classifier(X_train, X_test, y_train, y_test):
    TESTBATCH = 128

    print_blue("正在使用 BERT 进行分类...")
    if cfg.emb_method != "bert":
        raise ValueError(f"不支持的嵌入方法: {cfg.emb_method}")
    from transformers import BertForSequenceClassification, Trainer, TrainingArguments
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    
    # X_train: {'input_ids': tensor
    #            'token_type_ids': tensor
    #            'attention_mask': tensor}
    assert all(key in X_train for key in ["input_ids", "attention_mask"]), "X_train 必须包含 input_ids 和 attention_mask"

    from bert_dataset import BertDataset
    train_dataset = BertDataset(X_train["input_ids"], X_train["attention_mask"], torch.tensor(y_train))
    test_dataset = BertDataset(X_test["input_ids"], X_test["attention_mask"], torch.tensor(y_test))

    test_dataset.length = test_dataset.length // 10

    if cfg.cls_ldcpt:
        try:
            # load a checkpoint from cfg.logdir
            model = BertForSequenceClassification.from_pretrained(f"{cfg.logdir}/{cfg.emb_method}-{cfg.cls_method}")
            print_yellow("加载checkpoint成功")
        except Exception as e:
            print_yellow("加载checkpoint失败")
            raise e
    else:
        # 初始化BERT模型
        model = BertForSequenceClassification.from_pretrained('pretrained/bert-base-uncased', num_labels=2)
    model = model.to(cfg.device)
    
    # 训练参数设置
    training_args = TrainingArguments(
        max_steps=300, 
        per_device_train_batch_size=16,  # 较小batch size
        warmup_steps=100,  

        per_device_eval_batch_size=TESTBATCH,
        
        weight_decay=0.01,

        output_dir=f"{cfg.logdir}/{cfg.emb_method}-{cfg.cls_method}_output",
        logging_dir=f"{cfg.logdir}/{cfg.emb_method}-{cfg.cls_method}_log",
        logging_steps=50,  
        save_steps=100,    
        

        no_cuda=False,   
        fp16=cfg.half,   

        # evaluation_strategy="epoch",  # 每轮结束后评估
        # save_strategy="epoch",       # 每轮结束后保存模型
        # load_best_model_at_end=True, # 训练完成后加载最佳模型
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )
    
    # 训练模型
    trainer.train()

    # 保存
    model.save_pretrained(f"{cfg.logdir}/{cfg.emb_method}-{cfg.cls_method}")
    print_green("模型已保存。")


    # results = trainer.evaluate(test_dataset)
    results = {}
    test_loader = DataLoader(test_dataset, batch_size=TESTBATCH, shuffle=False)
    
    model.eval()  # 设置为评估模式
    y_pred = []
    y_proba = []
    
    with torch.no_grad():
        for batch in test_loader:
            # 将数据移动到设备上
            inputs = {
                'input_ids': batch[0].to(cfg.device),
                'attention_mask': batch[1].to(cfg.device)
            }
            
            outputs = model(**inputs)
            logits = outputs.logits
            logits_t2n = logits.cpu().numpy()
            batch_pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(batch_pred.tolist())
            y_proba.extend(logits_t2n.tolist())
    
    results = trainer.evaluate(test_dataset)
    print_green("BERT 分类完成。")

    results['y_pred'] = y_pred
    results['y_proba'] = y_proba
    results['y_test'] = y_test

    return results

def random_forest_classifier(X_train, X_test, y_train, y_test):
    print_blue("正在使用 RandomForest 进行分类...")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    model_path = f"{cfg.logdir}/{cfg.emb_method}-{cfg.cls_method}.pkl"
    
    if cfg.cls_ldcpt:
        try:
            # 从checkpoint加载
            with open(model_path, 'rb') as f:
                clf = pickle.load(f)
            print_yellow("加载checkpoint成功")
        except Exception as e:
            print_yellow("加载checkpoint失败")
            raise e
    else:
        clf = RandomForestClassifier(n_estimators=100, n_jobs=cfg.NJ, random_state=cfg.rs)
        clf.fit(X_train, y_train)
    
    # 预测并评估
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)  # 概率
    y_proba = [p[1] for p in y_proba]  # 正类概率
    accuracy = accuracy_score(y_test, y_pred)
    print_green("RandomForest 分类完成。")

    if not cfg.cls_ldcpt:
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)
        print_green("模型已保存。")

    return {
        'y_pred': y_pred,
        'y_proba': y_proba,
        'y_test': y_test,
        'accuracy': accuracy
    }

def knn_classifier(X_train, X_test, y_train, y_test):
    print_blue("正在使用 KNN 进行分类...")
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score
    
    model_path = f"{cfg.logdir}/{cfg.emb_method}-{cfg.cls_method}.pkl"
    
    if cfg.cls_ldcpt:
        try:
            # 从checkpoint加载
            with open(model_path, 'rb') as f:
                clf = pickle.load(f)
            print_yellow("加载checkpoint成功")
        except Exception as e:
            print_yellow("加载checkpoint失败")
            raise e
    else:
        clf = KNeighborsClassifier(n_neighbors=5, n_jobs=cfg.NJ)
        clf.fit(X_train, y_train)
    
    # 预测并评估
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)  # 概率
    y_proba = [p[1] for p in y_proba]  # 正类概率
    accuracy = accuracy_score(y_test, y_pred)
    print_green("KNN 分类完成。")

    if not cfg.cls_ldcpt:
        with open(model_path, 'wb') as f:
            pickle.dump(clf, f)
        print_green("模型已保存。")

    return {
        'y_pred': y_pred,
        'y_proba': y_proba,
        'y_test': y_test,
        'accuracy': accuracy
    }

def classify(X_train, X_test, y_train, y_test):
    if cfg.cls_method == "rf":
        return random_forest_classifier(X_train, X_test, y_train, y_test)
    elif cfg.cls_method == "knn":
        return knn_classifier(X_train, X_test, y_train, y_test)
    elif cfg.cls_method == "bert":
        return bert_classifier(X_train, X_test, y_train, y_test)
    else:
        raise ValueError(f"不支持的分类方法: {cfg.cls_method}")