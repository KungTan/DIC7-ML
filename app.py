import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import io
import time

# -- Page Configuration --
st.set_page_config(
    page_title="CRISP-DM Linear Regression Demo",
    page_icon="📈",
    layout="wide"
)

# -- Custom CSS for Premium Look --
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .crisp-header {
        color: #1e3d59;
        border-bottom: 2px solid #ff6e40;
        padding-bottom: 5px;
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .metric-container {
        display: flex;
        justify-content: space-between;
    }
</style>
""", unsafe_allow_html=True)

# -- Sidebar: Parameters --
st.sidebar.header("🛠️ 數據模擬參數")
n_samples = st.sidebar.slider("樣本數 (n)", 100, 1000, 500)
a_true = st.sidebar.slider("真實斜率 (a)", -10.0, 10.0, 2.5)
b_true = st.sidebar.slider("真實截距 (b)", -50.0, 50.0, 10.0)
noise_mean = st.sidebar.slider("雜訊平均值", -10.0, 10.0, 0.0)
noise_var = st.sidebar.slider("雜訊變異數", 0, 1000, 100)
random_seed = st.sidebar.number_input("隨機種子", value=42)

generate_btn = st.sidebar.button("🚀 生成數據並訓練模型")

# -- Helper Functions with Caching --
@st.cache_data
def get_synthetic_data(n, a, b, n_mean, n_var, seed):
    np.random.seed(seed)
    X = np.random.uniform(-100, 100, (n, 1))
    noise = np.random.normal(n_mean, np.sqrt(n_var), (n, 1))
    y = a * X + b + noise
    return X, y

# Initialize session state for data
if 'data_X' not in st.session_state or generate_btn:
    X, y = get_synthetic_data(n_samples, a_true, b_true, noise_mean, noise_var, random_seed)
    st.session_state['data_X'] = X
    st.session_state['data_y'] = y
    st.session_state['a_true'] = a_true
    st.session_state['b_true'] = b_true

# -- Main Page --
st.title("📊 線性回歸分析流程 (CRISP-DM)")
st.markdown("---")

X = st.session_state['data_X']
y = st.session_state['data_y']
a_true = st.session_state['a_true']
b_true = st.session_state['b_true']

# Phase 1: Business Understanding
st.markdown("<h2 class='crisp-header'>1. 業務理解 (Business Understanding)</h2>", unsafe_allow_html=True)
st.info("目標：根據特徵 X 預測連續型變數 Y。此階段確認數據挖掘的目標，並制定實施方案。")

# Phase 2: Data Understanding
st.markdown("<h2 class='crisp-header'>2. 數據理解 (Data Understanding)</h2>", unsafe_allow_html=True)
col1, col2 = st.columns([1, 2])
with col1:
    st.subheader("數據概覽")
    df_raw = pd.DataFrame(X, columns=['Feature_X'])
    df_raw['Target_Y'] = y
    st.write(df_raw.describe())
with col2:
    st.subheader("分佈圖")
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    sns.scatterplot(x='Feature_X', y='Target_Y', data=df_raw, alpha=0.5, ax=ax1)
    ax1.set_title("Generated Data Distribution")
    st.pyplot(fig1)

# Phase 3: Data Preparation
st.markdown("<h2 class='crisp-header'>3. 數據準備 (Data Preparation)</h2>", unsafe_allow_html=True)
with st.spinner("正在進行數據預處理..."):
    X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), test_size=0.2, random_state=random_seed)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    st.success(f"數據準備就緒：訓練集大小 {len(X_train)}，測試集大小 {len(X_test)}。已套用 StandardScaler。")

# Phase 4: Modeling
st.markdown("<h2 class='crisp-header'>4. 建模 (Modeling)</h2>", unsafe_allow_html=True)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Derive original scale parameters
# y = m_scaled * ( (X - mean)/std ) + b_scaled
# y = (m_scaled/std) * X + (b_scaled - m_scaled*mean/std)
learned_a = model.coef_[0] / scaler.scale_[0]
learned_b = model.intercept_ - (model.coef_[0] * scaler.mean_[0] / scaler.scale_[0])

st.write("### 模型參數對照")
comp_data = {
    "參數": ["斜率 (Slope)", "截距 (Intercept)"],
    "真實設定值": [a_true, b_true],
    "模型學習值": [float(learned_a.item()), float(learned_b.item())]
}
st.table(pd.DataFrame(comp_data))

# Phase 5: Evaluation
st.markdown("<h2 class='crisp-header'>5. 評估 (Evaluation)</h2>", unsafe_allow_html=True)
y_pred = model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

m1, m2, m3 = st.columns(3)
m1.metric("MSE", f"{mse:.2f}")
m2.metric("RMSE", f"{rmse:.2f}")
m3.metric("R²", f"{r2:.4f}")

st.subheader("回歸擬合結果")
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.scatter(X_test, y_test, color='gray', alpha=0.5, label='Test Data')
# Plot regression line
x_axis = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_axis_pred = model.predict(scaler.transform(x_axis))
ax2.plot(x_axis, y_axis_pred, color='red', linewidth=3, label='Regression Line')
ax2.set_xlabel("Feature X")
ax2.set_ylabel("Target Y")
ax2.legend()
st.pyplot(fig2)

# Phase 6: Deployment
st.markdown("<h2 class='crisp-header'>6. 部署 (Deployment)</h2>", unsafe_allow_html=True)
c_dep1, c_dep2 = st.columns(2)

with c_dep1:
    st.subheader("🔮 線上預測")
    user_x = st.number_input("輸入 X 值：", value=0.0)
    user_x_scaled = scaler.transform([[user_x]])
    prediction = model.predict(user_x_scaled)
    st.write(f"預測 Y 值為：**{float(prediction.item()):.4f}**")

with c_dep2:
    st.subheader("💾 模型儲存")
    model_data = {
        'model': model,
        'scaler': scaler,
        'metrics': {'mse': mse, 'r2': r2},
        'metadata': {'a_true': a_true, 'b_true': b_true}
    }
    
    buf = io.BytesIO()
    joblib.dump(model_data, buf)
    st.download_button(
        label="⬇️ 下載模型 (joblib)",
        data=buf.getvalue(),
        file_name="linear_model.joblib",
        mime="application/octet-stream"
    )

st.markdown("---")
st.caption("Powered by Scikit-Learn & Streamlit | CRISP-DM Workflow Demo")
