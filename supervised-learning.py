# 監督式學習
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 讀取資料
url = "https://raw.githubusercontent.com/v123582/edu-dataset/main/student-performance-dataset.csv"
df = pd.read_csv(url)

# 建立標籤
df['pass'] = (df['G3'] >= 10).astype(int)

# 選擇輸入欄位
features = ['G1', 'G2', 'studytime', 'failures', 'absences', 'famsup', 'schoolsup']
X = pd.get_dummies(df[features], drop_first=True)
y = df['pass']

# 切分資料
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 建立模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 預測與評估
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
