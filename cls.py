"""
    by shc 2025.5.3
"""
import os
import signal
import pickle
import shutil
import numpy as np
from UTIL.colorful import print_blue, print_green, print_yellow
from global_config import GlobalConfig as cfg
from sklearn.metrics import accuracy_score

logdir = cfg.taskdir

def bert_classifier(X_train, X_test, y_train, y_test):
    TESTBATCH = 128

    print_blue("正在使用 BERT 进行分类...")
    if cfg.emb_method != "bert":
        raise ValueError(f"不支持的嵌入方法: {cfg.emb_method}")
    
    from transformers import BertForSequenceClassification, Trainer, TrainingArguments
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    import numpy as np
    import os
    import signal
    from bert_dataset import BertDataset

    assert all(key in X_train for key in ["input_ids", "attention_mask"]), "X_train 必须包含 input_ids 和 attention_mask"

    # 数据集compat
    train_dataset = BertDataset(X_train["input_ids"], X_train["attention_mask"], torch.tensor(y_train))
    test_dataset = BertDataset(X_test["input_ids"], X_test["attention_mask"], torch.tensor(y_test))
    test_dataset.length = test_dataset.length // 10 

    # 模型路径
    os.makedirs(logdir, exist_ok=True)
    model_path = f"{logdir}/model"

    if cfg.cls_ldcpt:
        try:
            model = BertForSequenceClassification.from_pretrained(model_path)
            print_yellow("加载checkpoint成功")
        except Exception as e:
            print_yellow("加载checkpoint失败")
            raise e
    else:
        model = BertForSequenceClassification.from_pretrained('pretrained/bert-base-uncased', num_labels=2)
    
    model = model.to(cfg.device)

    # 参数
    training_args = TrainingArguments(
        max_steps=1000, 
        per_device_train_batch_size=16,
        warmup_steps=100,
        per_device_eval_batch_size=TESTBATCH,
        weight_decay=0.01,
        output_dir=f"{model_path}_output",
        logging_dir=f"{model_path}_log",
        logging_steps=50,
        save_steps=100,
        no_cuda=False,
        fp16=cfg.half,
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # 训练
    try:
        print_blue("开始训练 Ctrl+C可终止...")
        trainer.train()
    except KeyboardInterrupt:
        print_yellow("\n训练被中断")
        model.save_pretrained(model_path)
        print_green("模型已保存。")
    else:
        print_green("训练完成")
        model.save_pretrained(model_path) 
        print_green("模型已保存。")

    # 测试
    print_blue("正在测试模型...")
    test_loader = DataLoader(test_dataset, batch_size=TESTBATCH, shuffle=False)
    
    model.eval()  # 评估模式
    y_pred = []
    y_proba = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = {
                'input_ids': batch[0].to(cfg.device),
                'attention_mask': batch[1].to(cfg.device)
            }
            outputs = model(**inputs)
            logits = outputs.logits
            batch_pred = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(batch_pred.tolist())
            y_proba.extend(logits.cpu().numpy().tolist())
    
    # results = trainer.evaluate(test_dataset)
    results = {}
    print_green("BERT 分类完成。")

    results['y_pred'] = y_pred
    results['y_proba'] = y_proba
    results['y_test'] = y_test

    return results
    

def lstm_classifier(X_train, X_test, y_train, y_test):
    print_blue("正在使用 LSTM 进行分类...")
    if cfg.emb_method != "seqword2vec":
        raise ValueError(f"不支持的嵌入方法: {cfg.emb_method}")

    X_train = X_train
    X_test = X_test

    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
    from lstm import LSTMModel, train_model, pad_sequences
    
    # 参数设置
    # VOCAB_SIZE = len(word_to_idx) + 1  # +1 for padding (0)
    EMBEDDING_DIM = cfg.word2vec_emb_dim
    HIDDEN_DIM = 128
    BATCH_SIZE = 32
    TEST_BATCH_SIZE = 128
    EPOCHS = 100
    LR = 0.01
    os.makedirs(logdir, exist_ok=True)
    model_path = f"{logdir}/model.pt"
    rec_des = f"{logdir}/rec.jpg"
    rec_src = "./VISUALIZE_logdir/rec.jpg"
    

    # 确定最大序列长度
    max_seq_len = max(len(seq) for seq in X_train)
    min_seq_len = min(len(seq) for seq in X_train)
    print(f"最大序列长度: {max_seq_len}, 最小序列长度: {min_seq_len}")
    # X_train_pad = pad_sequences(X_train, maxlen=max_seq_len)
    # X_test_pad = pad_sequences(X_test, maxlen=max_seq_len)
    
    # 转换为Tensor
    train_data = TensorDataset(torch.Tensor(X_train).float(), torch.Tensor(y_train))
    test_data = TensorDataset(torch.Tensor(X_test).float(), torch.Tensor(y_test))
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False)
    
    model = LSTMModel(EMBEDDING_DIM, HIDDEN_DIM, num_layers=1, num_classes=2).to(cfg.device)
    
    
    # 加载checkpoint
    if cfg.cls_ldcpt and os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            print_yellow("加载checkpoint成功")
        except Exception as e:
            print_yellow(f"加载checkpoint失败: {str(e)}")
    
    # 训练
    try:
        print_blue("开始训练 (Ctrl+C可终止)...")
        train_model(model, train_loader, test_loader, EPOCHS, LR)
    except KeyboardInterrupt:
        print_yellow("\n训练被中断")
        torch.save(model.state_dict(), model_path)
        if os.path.exists(rec_src):
            shutil.copy(rec_src, rec_des)
        print_green("模型已保存。")
    else:
        print_green("训练完成")
        torch.save(model.state_dict(), model_path)
        if os.path.exists(rec_src):
            shutil.copy(rec_src, rec_des)
        print_green("模型已保存。")
    
    # 测试
    print_blue("正在测试模型...")
    model.eval()
    y_pred = []
    y_proba = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(cfg.device)
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            y_proba.extend(probabilities[:, 1].cpu().numpy().tolist())
            y_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy().tolist())
    
    accuracy = accuracy_score(y_test, y_pred)
    print_green(f"LSTM 分类完成，准确率: {accuracy:.4f}")
    
    return {
        'y_pred': y_pred,
        'y_proba': y_proba,
        'y_test': y_test,
        'accuracy': accuracy
    }

def random_forest_classifier(X_train, X_test, y_train, y_test):
    print_blue("正在使用 RandomForest 进行分类...")
    from sklearn.ensemble import RandomForestClassifier

    os.makedirs(logdir, exist_ok=True)
    model_path = f"{logdir}/model.pkl"
    
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
    
    os.makedirs(logdir, exist_ok=True)
    model_path = f"{logdir}/model.pkl"
    
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
    elif cfg.cls_method == "lstm":
        return lstm_classifier(X_train, X_test, y_train, y_test)
    else:
        raise ValueError(f"不支持的分类方法: {cfg.cls_method}")