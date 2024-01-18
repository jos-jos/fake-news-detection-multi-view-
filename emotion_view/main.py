import time
from datetime import timedelta
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd

from data_preprocess import load_imdb
from utils import set_seed
from model import TextCNN

# 加快运算速度，需要GPU
set_seed()

# 设置参数
BATCH_SIZE = 50
# BATCH_SIZE = 64
LEARNING_RATE = 0.0009
NUM_EPOCHS = 8
TEST_NUM = 1


def get_time_dif(start_time):
    """Gets used time"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def calculate_metrics(TP, FP, FN, TN):
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP != 0 else 0
    recall = TP / (TP + FN) if TP + FN != 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    return accuracy, precision, recall, f1_score
all_predictions = []
start_time = time.time()

train_data, test_data, vocab = load_imdb()
# test_data, vocab = load_imdb()
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)


# for text, label in train_loader:
#     print(f'text.shape{text.shape}') #label.shapetorch.Size([50])

#     print(f'label.shape{label.shape}') #text.shapetorch.Size([50, 512])

device = 'cuda' if torch.cuda.is_available() else 'cpu'  # GPU训练
model = TextCNN(vocab).to(device)
# print(model)
criterion = nn.CrossEntropyLoss()  # 获取损失函数
optimizer = torch.optim.Adam (model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4)  # 获取优化器
# model.load_state_dict(torch.load("./200-0.91.pth",map_location = torch.device('cpu')))

for epoch in range(1, NUM_EPOCHS + 1):
    print(f'Epoch {epoch}\n' + '-' * 32)
    avg_train_loss = 0
    TP, FP, FN, TN = 0, 0, 0, 0
    for batch_idx, (text, label) in enumerate(train_loader):
        # 前向计算->计算损失函数->(从损失函数)反向传播->更新网络
        text, label = text.to(device), label.to(device)
        predicted_label = model(text)
        loss = criterion(predicted_label, label)
        avg_train_loss += loss
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 反向传播计算梯度
        optimizer.step()  # 更新网络参数

        # Calculate metrics
        predicted_classes = predicted_label.argmax(1)
        TP += ((predicted_classes == 1) & (label == 1)).sum().item()
        FP += ((predicted_classes == 1) & (label == 0)).sum().item()
        FN += ((predicted_classes == 0) & (label == 1)).sum().item()
        TN += ((predicted_classes == 0) & (label == 0)).sum().item()
        if (batch_idx + 1) % 5 == 0:
            print(f"[{(batch_idx + 1) * BATCH_SIZE:>5}/{len(train_loader.dataset):>5}] train loss: {loss:.4f}")
            # get_epoch = epoch
            # get_step = (batch_idx + 1) * BATCH_SIZE
            # get_loss = loss
            # list = [get_epoch, get_step, get_loss]
            # data = pd.DataFrame([list])
            # data.to_csv('train_acc.csv', mode='a', header=False, index=False)

    print(f"Avg train loss: {avg_train_loss / (batch_idx + 1):.4f}\n")

    time_dif = get_time_dif(start_time)
    print("Time Usage:", time_dif)

    # 保存模型
    torch.save(model, 'textcnn.pt')

    # 进行测试
    if epoch % TEST_NUM == 0:
        acc = 0
        TP_test, FP_test, FN_test, TN_test = 0, 0, 0, 0
        for X, y in test_loader:
            with torch.no_grad():
                X, y = X.to(device), y.to(device)
                pred = model(X)
                acc += (pred.argmax(1) == y).sum().item()
                predicted_classes = pred.argmax(1)

                TP_test += ((predicted_classes == 1) & (y == 1)).sum().item()
                FP_test += ((predicted_classes == 1) & (y == 0)).sum().item()
                FN_test += ((predicted_classes == 0) & (y == 1)).sum().item()
                TN_test += ((predicted_classes == 0) & (y == 0)).sum().item()

        accuracy, precision, recall, f1_score = calculate_metrics(TP_test, FP_test, FN_test, TN_test)
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1_score:.4f}")
        print(f"Accuracy: {acc / len(test_loader.dataset):.4f}")


# for X, y in test_loader:
#             with torch.no_grad():
#                 X, y = X.to(device), y.to(device)
#                 pred = model(X)
#                 # acc += (pred.argmax(1) == y).sum().item()
#                 print(pred)
#                 all_predictions.append(pred)

# df = pd.read_csv('./datasets/style_.csv')
all_predictions_tensor = torch.cat(all_predictions, dim=0)

# Convert to numpy and then to list
all_predictions_list = all_predictions_tensor.cpu().numpy().tolist()

# Add to DataFrame
# df['emotion_feature'] = all_predictions_list
# df.to_csv('./datasets/sty_emo_.csv')
# # Just showing the first few rows of the updated DataFrame for demonstration
# print(df.head)
