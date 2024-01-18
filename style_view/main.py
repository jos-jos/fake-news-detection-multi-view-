import time
from datetime import timedelta
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from data_preprocess import load_imdb
from utils import set_seed
from model import MLP

# 加快运算速度，需要GPU
set_seed()

# 设置参数
BATCH_SIZE = 50
LEARNING_RATE = 0.0009
NUM_EPOCHS = 40


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


start_time = time.time()

train_data, test_data, vocab = load_imdb()
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # GPU训练
model = MLP(vocab).to(device)
# print(model)
criterion = nn.L1Loss(reduction="sum")  # 获取损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=3e-3)  # 获取优化器
model.load_state_dict(torch.load('model/mlp_saved.pth'))

for epoch in range(1, NUM_EPOCHS + 1):
    print(f'Epoch {epoch}\n' + '-' * 32)
    avg_train_loss = 0
    for batch_idx, (text, label) in enumerate(train_loader):
        # 前向计算->计算损失函数->(从损失函数)反向传播->更新网络
        text, label = text.to(device), label.to(device)
        predicted_label = model(text) - 1.0
        loss = criterion(predicted_label, label)
        avg_train_loss += loss
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新网络参数

        if (batch_idx + 1) % 5 == 0:
            torch.save(model.state_dict(), 'model/mlp.pth')
            print(f"[{(batch_idx + 1) * BATCH_SIZE:>5}/{len(train_loader.dataset):>5}] train loss: {loss/BATCH_SIZE/8:.4f}")
            # get_epoch = epoch
            # get_step = (batch_idx + 1) * BATCH_SIZE
            # get_loss = loss
            # list = [get_epoch, get_step, get_loss]
            # data = pd.DataFrame([list])
            # data.to_csv('train_acc.csv', mode='a', header=False, index=False)
    print(f"Avg train loss: {avg_train_loss / (batch_idx + 1)/BATCH_SIZE/8:.4f}\n")

    avg_test_loss = 0
    for X, y in test_loader:
        with torch.no_grad():
            X, y = X.to(device), y.to(device)
            pred = model(X) - 1.0
            loss = criterion(pred, y)
            avg_test_loss += loss
    print(f"Avg test loss: {avg_train_loss / (batch_idx + 1) / BATCH_SIZE/8 /2 * 100:.4f}%\n")



time_dif = get_time_dif(start_time)
print("Time Usage:", time_dif)

