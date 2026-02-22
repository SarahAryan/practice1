import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import os
import warnings

from google.colab import drive
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')
warnings.filterwarnings('ignore')
input_file = "/content/drive/MyDrive/sample_records.csv"
outcome_col = "metadata.conclusion"

if not os.path.exists(input_file):
    raise FileNotFoundError(f"فایل در مسیر {input_file} یافت نشد!")

df = pd.read_csv(input_file)
df = df.dropna(subset=[outcome_col])

print("--- گزارش وضعیت اولیه داده‌ها ---")
print(df[outcome_col].value_counts())
print("--------------------------------")

df['target'] = df[outcome_col].apply(lambda x: 1 if str(x).lower() == 'failure' else 0)

highlighted_features = ["repository_name", "metadata.workflow_id", "metadata.event", "metadata.actor.login"]
label_encoders = {}
for col in highlighted_features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].fillna('MISSING').astype(str))
    label_encoders[col] = le

df = df.sort_values(by=["repository_name", "metadata.run_started_at"])
X_seq, y_seq = [], []

for repo, group in df.groupby("repository_name"):
    features = group[highlighted_features].values.tolist()
    targets = group['target'].values.tolist()
    for i in range(1, len(features)):
        X_seq.append(features[:i])
        y_seq.append(targets[i])

y_seq = np.array(y_seq)
print(f"\nتعداد کل توالی‌های ساخته شده: {len(y_seq)}")
print(f"تعداد شکست‌ها در توالی‌ها: {sum(y_seq)}")

if sum(y_seq) < 2:
    print("⚠️ خطا: تعداد شکست‌ها برای آموزش و تست کافی نیست! (حداقل ۲ مورد نیاز است)")
else:
    # تقسیم استراتژیک (Stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_seq, test_size=0.3, random_state=42, stratify=y_seq
    )

    print(f"تعداد شکست در آموزش: {sum(y_train)}")
    print(f"تعداد شکست در تست: {sum(y_test)}")

    pos_weight = torch.tensor([(len(y_train) - sum(y_train)) / (sum(y_train) + 1e-6)])

    class SequenceDataset(Dataset):
        def __init__(self, X, y):
            self.X = [torch.tensor(seq, dtype=torch.float32) for seq in X]
            self.y = torch.tensor(y, dtype=torch.float32)
        def __len__(self): return len(self.y)
        def __getitem__(self, idx): return self.X[idx], self.y[idx]

    def collate_fn(batch):
        seqs, labels = zip(*batch)
        return torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True), torch.stack(labels)

    train_loader = DataLoader(SequenceDataset(X_train, y_train), batch_size=4, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(SequenceDataset(X_test, y_test), batch_size=1, collate_fn=collate_fn)

    class StrongLSTM(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, 128, num_layers=2, batch_first=True, dropout=0.2)
            self.fc = nn.Linear(128, 1)
        def forward(self, x):
            _, (hn, _) = self.lstm(x)
            return self.fc(hn[-1])

    model = StrongLSTM(len(highlighted_features))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    for epoch in range(100):
        model.train()
        for b_x, b_y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(b_x).squeeze(1), b_y)
            loss.backward()
            optimizer.step()

    
    model.eval()
    preds, actuals = [], []
    with torch.no_grad():
        for b_x, b_y in test_loader:
            prob = torch.sigmoid(model(b_x)).item()
            preds.append(1 if prob > 0.5 else 0)
            actuals.append(int(b_y.item()))

    print("\n" + "="*30)
    print("نتایج نهایی:")
    print(confusion_matrix(actuals, preds))
    print(classification_report(actuals, preds, target_names=['Success', 'Failure']))
    print("="*30)