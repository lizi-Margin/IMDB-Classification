import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from VISUALIZE.mcv_log_manager import LogManager
from global_config import GlobalConfig as cfg
class LSTMModel(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(emb_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])  # 只取最后一个时间步的输出
        out = self.softmax(out)
        return out

def train_model(model, train_loader, test_loader, epochs, lr):
        lm = LogManager(who="lstm.py")
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        
        
        # use tqdm to show progress bar
        # for epoch in tqdm(range(epochs), desc="Training"):
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            total_acc = 0
            subprogress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)
            for batch_idx, (data, target) in enumerate(subprogress):
                data, target = data.float().to(cfg.device), target.float().to(cfg.device)
                # print(data.shape, target.shape)
                # print(torch.max(data), torch.min(data), torch.mean(data))
                # print(torch.max(target), torch.min(target), torch.mean(target))
                
                optimizer.zero_grad()
                output = model(data)
                output_index = torch.argmax(output, dim=1)
                output = output[:, 1]
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                accuracy = torch.sum(output_index.detach() == target).item() / target.size(0)
                total_acc += accuracy
                total_loss += loss.item()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")

            info = {
                "epoch":epoch+1,
                "loss":total_loss/len(train_loader),
                'accuracy': total_acc/len(train_loader),
            }


            if epoch % 5 == 0:
                model.eval()
                test_loss = 0
                test_acc = 0
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.float().to(cfg.device), target.float().to(cfg.device)
                        output = model(data)
                        output_index = torch.argmax(output, dim=1)
                        output = output[:, 1]
                        loss = criterion(output, target)
                        accuracy = torch.sum(output_index.detach() == target).item() / target.size(0)
                        test_acc += accuracy
                        test_loss += loss.item()

                info.update({
                    "test_loss":test_loss/len(test_loader),
                    "test_accuracy":test_acc/len(test_loader),
                })
            lm.log_trivial(info)
            lm.log_trivial_finalize()

def pad_sequences(sequences, maxlen=None, padding='post', value=0):
    if maxlen is None:
        minlen = min(len(seq) for seq in sequences)
        maxlen = max(len(seq) for seq in sequences)
        meanlen = int(np.mean([len(seq) for seq in sequences]))
        print(f"[pad_sequences] minlen: {minlen}, maxlen: {maxlen}, meanlen: {meanlen}")
    print(f"[pad_sequences] use maxlen: {maxlen}")
    
    padded = []
    # use tqdm
    for seq in tqdm(sequences, desc="Padding"):
        if len(seq) >= maxlen:
            padded_seq = seq[:maxlen]
        else:
            pad_len = maxlen - len(seq)
            padded_seq = seq + [value] * pad_len if padding == 'post' else [value] * pad_len + seq
        padded.append(padded_seq)
        # print(repr(padded_seq))
        # print(len(padded_seq))
        # for e in padded_seq:
        #     # print(e)
        #     print(len(e))
        # exit(1)
    return np.array(padded)
    