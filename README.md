# Machine Learning Analysis on Student Performance Dataset

本專題以 **Student Performance Dataset** 為研究對象，透過機器學習方法分析學生學習表現，分別建立一個 **監督式學習模型** 與一個 **非監督式學習模型**，以探討學生是否期末及格，以及學生特徵之間的潛在分群結構。

---

## Dataset Description

- 資料來源：Student Performance Dataset（UCI Machine Learning Repository）
- 資料內容：包含學生背景、學習習慣、家庭因素與成績（G1、G2、G3）
- 總欄位數：33
- 資料筆數：約 395 筆

---

## Supervised Learning（監督式學習）

### Problem Definition
- 任務類型：二元分類（Binary Classification）
- 目標：預測學生是否「期末及格」
- Label 定義：
  - `pass = 1` ：G3 ≥ 10
  - `pass = 0` ：G3 < 10

### Input Features
- G1、G2
- studytime
- failures
- absences
- famsup
- schoolsup

（類別型資料使用 One-Hot Encoding 處理）

### Model
- Logistic Regression
- 資料切分比例：Training / Testing = 8 : 2
- 評估指標：
  - Accuracy
  - Precision / Recall
  - F1-score

### Code
- 檔案：`supervised-learning.py`

---

## Unsupervised Learning（非監督式學習）

### Problem Definition
- 任務類型：Clustering（分群）
- 目標：在無標籤情況下，找出學生學習表現的潛在結構

### Input Features
- G1、G2、G3
- studytime
- failures
- absences

### Model
- K-Means Clustering
- 群數設定：k = 3
- 前處理：
  - StandardScaler 標準化數值特徵
- 視覺化：
  - 以 G1 與 G3 繪製分群結果散佈圖

### Observation
- 大致可區分為低成績、中等成績與高成績群
- 部分低 G3 分數點被分到高風險群，可能原因包括：
  - K-Means 以距離為依據，未考慮成績的時間順序
  - G3 與其他特徵（如缺席、failures）在距離空間中權重影響

### Code
- 檔案：`unsupervised-learning.py`

---

## Conclusion

- G1、G2 對於預測期末成績具有高度影響力，適合作為早期預警指標
- 監督式學習能有效預測學生是否及格，但需注意類別不平衡問題
- 非監督式學習有助於探索學生族群結構，但分群結果需搭配教育背景解讀
- 若應用於實務教育決策，建議結合模型解釋性與公平性分析

---

## How to Run

```bash
python supervised-learning.py
python unsupervised-learning.py
