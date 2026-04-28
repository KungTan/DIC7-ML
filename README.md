# CRISP-DM 線性回歸演示工具 (Linear Regression Demo)

這是一個基於 **Streamlit** 與 **Scikit-Learn** 開發的互動式 Web 應用程式，旨在向使用者演示如何遵循數據科學標準開發流程 **CRISP-DM** 進行數據模擬、模型訓練與評估。

### 🌐 Live Demo
👉 [https://dic7-ml.streamlit.app/](https://dic7-ml.streamlit.app/)

## 🌟 功能特點
- **CRISP-DM 全流程展示**：完整包含從「業務理解」到「部署」的六個標準階段。
- **互動式數據模擬**：透過側邊欄滑桿即時生成自定義樣本數、斜率、截距及雜訊。
- **自動化數據預處理**：內建 `train_test_split` 數據分割與 `StandardScaler` 標準化流程。
- **模型訓練與比較**：使用 `LinearRegression` 建模，並將模型學習到的參數與原始真實設定值進行直觀對照。
- **效能評估指標**：即時計算 MSE、RMSE 與 R² 分數，並繪製專業的回歸擬合圖。
- **線上預測與匯出**：提供互動式輸入框進行即時預測，並支援下載訓練好的 `.joblib` 模型檔。

## 🚀 快速開始

### 1. 安裝環境依賴
確保你已安裝 Python (建議 3.9+)，然後在終端機執行：
```bash
pip install -r requirements.txt
```

### 2. 啟動應用程式
在專案根目錄下執行以下指令：
```bash
python -m streamlit run app.py
```
啟動後，瀏覽器會自動開啟 [http://localhost:8501](http://localhost:8501)。

## 流程說明 (CRISP-DM Phases)
1. **業務理解 (Business Understanding)**：定義預測目標。
2. **數據理解 (Data Understanding)**：觀察生成數據的分佈與統計特徵。
3. **數據準備 (Data Preparation)**：特徵縮放與數據分割。
4. **建模 (Modeling)**：線性回歸演算法擬合。
5. **評估 (Evaluation)**：檢查誤差指標與擬合效果。
6. **部署 (Deployment)**：應用模型進行預測。

## 專案結構
- `app.py`: 主程式碼，包含所有介面邏輯與機器學習運算。
- `requirements.txt`: 專案所需的庫清單。
- `chat_history.md`: 本專案的開發歷程記錄。

---
*Powered by Antigravity AI*
