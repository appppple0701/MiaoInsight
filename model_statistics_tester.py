import pandas as pd
import numpy as np
import joblib
from scipy import stats

def generate_detailed_report():
    # 1. 載入模型殘差 (s)
    try:
        reg_s = joblib.load('registration_residual_std.pkl')
        open_s = joblib.load('open_residual_std.pkl')
    except:
        print("錯誤：找不到 .pkl 檔案。")
        return

    # 2. 定義 7 個模型的測試數據
    # 請在此更新你實際得到的預測與觀測數據
    test_data = [
        {"name": "歌唱比賽", "type": "Reg", "pred": 120, "actual": 115},
        {"name": "書法活動", "type": "Reg", "pred": 85, "actual": 92},
        {"name": "天赦日", "type": "Reg", "pred": 1500, "actual": 1480},
        {"name": "乞龜", "type": "Open", "pred": 3200, "actual": 3100},
        {"name": "關公誕辰", "type": "Open", "pred": 5000, "actual": 5200},
        {"name": "盂蘭盆節", "type": "Open", "pred": 2800, "actual": 2750},
        {"name": "新春團拜", "type": "Open", "pred": 1200, "actual": 1350}
    ]

    alpha = 0.01
    n = 30  # 樣本數基準
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha/2, df)

    final_results = []

    for item in test_data:
        # 根據模型類型選擇對應的殘差標準差
        s = reg_s if item["type"] == "Reg" else open_s
        se = s / np.sqrt(n)
        
        # 計算 t 統計量與 P-value
        t_stat = (item["actual"] - item["pred"]) / se
        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df))
        
        # 計算信賴區間 (CI)，並修正下界不為負數
        raw_lower = item["pred"] - t_crit * se
        ci_lower = max(0, raw_lower) 
        ci_upper = item["pred"] + t_crit * se
        
        final_results.append({
            "模型項目": item["name"],
            "H0 (預測 μ)": item["pred"],
            "實際觀測值": item["actual"],
            "99% CI 下界": round(ci_lower, 2),
            "99% CI 上界": round(ci_upper, 2),
            "P-value": f"{p_val:.4f}",
            "CR 臨界值": f"±{t_crit:.3f}",
            "判定": "顯著偏離" if p_val < alpha else "正常範圍"
        })

    # 3. 產出表格
    df_report = pd.DataFrame(final_results)
    
    # 控制台輸出
    print("\n" + "="*85)
    print(f" 專案測試細節報告：7大模型數據檢定 (信心水準 99%) ")
    print("="*85)
    print(df_report.to_string(index=False))
    
    # 導出 CSV
    df_report.to_csv("model_test_report.csv", index=False, encoding="utf-8-sig")
    print("\n[完成] 詳細報表已儲存至: model_test_report.csv")

if __name__ == "__main__":
    generate_detailed_report()