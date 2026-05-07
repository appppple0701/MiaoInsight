from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from fastapi.responses import HTMLResponse

app = FastAPI()

# ======================================
# CORS 設定 (允許前後端跨埠溝通)
# ======================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ======================================
# 載入模型與資料
# ======================================
# 確保這些檔案與 app.py 放在同一個根目錄
model = joblib.load("event_model.pkl")
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv").values.ravel()
FEATURE_COLUMNS = list(X_train.columns)

# ======================================
# 資料格式定義
# ======================================
class EventInput(BaseModel):
    FB_exposure: int
    IG_exposure: int
    YT_exposure: int
    Ad_spend: int
    duration_hours: int
    prize_value: int
    num_staff: int
    queue_design_score: float
    is_weekend: int
    is_holiday: int
    weather_score: float
    past_avg: int
    past_max: int

# ======================================
# 策略優化邏輯
# ======================================
def optimize_event(data):
    best_config = data.copy()
    # 模擬優化建議：增加 20% 曝光與 10% 預算
    best_config['FB_exposure'] = int(data['FB_exposure'] * 1.2)
    best_config['Ad_spend'] = int(data['Ad_spend'] * 1.1)
    return {
        "best_config": best_config,
        "best_config_prediction": int(5500) # 模擬優化後的數值
    }

# ======================================
# API 路由 (資料交換)
# ======================================
@app.post("/predict")
def predict(data: EventInput):
    data_dict = data.dict()
    df = pd.DataFrame([data_dict])[FEATURE_COLUMNS]
    
    # 執行預測
    pred = int(model.predict(df)[0])
    
    # 執行優化分析
    roi_result = optimize_event(data_dict)
    
    # 回傳給前端 (欄位名稱與 simple_mode.html 的 JS 對應)
    return {
        "prediction": pred,
        "roi_optimization": roi_result, 
        "feature_importance": {
            "社群曝光": 0.4, 
            "廣告預算": 0.3, 
            "人員配置": 0.2, 
            "天氣與時間": 0.1
        }
    }

# ======================================
# 網頁路由 (服務靜態 HTML 檔案)
# ======================================

# 1. 首頁 (index.html)
@app.get("/", response_class=HTMLResponse)
def get_index():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "前端", "index.html")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h3>錯誤：找不到 index.html，請檢查『前端』資料夾位置。</h3>"

# 2. 簡易模式頁面 (simple_mode.html)
@app.get("/simple", response_class=HTMLResponse)
def get_simple_mode():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "前端", "simple_mode.html")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h3>錯誤：找不到 simple_mode.html，請檢查檔案名稱。</h3>"

# 3. 建議模式頁面 (idea_mode.html) - 預留擴充
@app.get("/idea", response_class=HTMLResponse)
def get_idea_mode():
    base_path = os.path.dirname(__file__)
    file_path = os.path.join(base_path, "前端", "idea_mode.html")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h3>錯誤：找不到 idea_mode.html。</h3>"