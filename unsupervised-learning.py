# 非監督式學習
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 讀取資料
url = "https://raw.githubusercontent.com/v123582/edu-dataset/main/student-performance-dataset.csv"
df = pd.read_csv(url)

# 選擇分群欄位
features = ['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']
X = df[features]

# 標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means 分群
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)

df['cluster'] = labels

# 視覺化
plt.scatter(df['G1'], df['G3'], c=labels)
plt.xlabel('G1')
plt.ylabel('G3')
plt.title('K-Means Clustering Result')
plt.show()
