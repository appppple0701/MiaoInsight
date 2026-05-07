import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.inspection import permutation_importance

# ======================
#  產生資料（保留）
# ======================
np.random.seed(42)

data = pd.DataFrame({
    'FB_exposure': np.random.randint(500, 5000, 300),
    'IG_exposure': np.random.randint(500, 4000, 300),
    'YT_exposure': np.random.randint(200, 2000, 300),
    'Ad_spend': np.random.randint(5000, 50000, 300),

    'duration_hours': np.random.randint(2, 10, 300),
    'prize_value': np.random.randint(1000, 20000, 300),
    'num_staff': np.random.randint(5, 50, 300),
    'queue_design_score': np.random.uniform(0.5, 1.0, 300),

    'is_weekend': np.random.randint(0, 2, 300),
    'is_holiday': np.random.randint(0, 2, 300),
    'weather_score': np.random.uniform(0.6, 1.0, 300),

    'past_avg': np.random.randint(1000, 3000, 300),
    'past_max': np.random.randint(1500, 4000, 300),
})

data['attendance'] = (
    0.3*data['FB_exposure'] +
    0.5*data['IG_exposure'] +
    0.2*data['YT_exposure'] +
    0.05*data['Ad_spend'] +
    20*data['duration_hours'] +
    0.01*data['prize_value'] +
    15*data['num_staff'] +
    500*data['queue_design_score'] +
    300*data['is_holiday'] +
    200*data['is_weekend'] +
    400*data['weather_score'] +
    0.5*data['past_avg'] +
    np.random.normal(0, 200, 300)
)

# ======================
# 資料前處理
# ======================
X = data.drop(columns=['attendance'])
y = data['attendance']

# 缺失值處理（保險）
X = X.fillna(X.median())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# ======================
#  Pipeline + Random Search
# ======================
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor())
])

param_dist = {
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [2, 3, 4],
    'model__learning_rate': [0.01, 0.05, 0.1],
}

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_dist,
    n_iter=10,
    cv=3,
    scoring='neg_mean_squared_error',
    random_state=42
)

search.fit(X_train, y_train)
model = search.best_estimator_

# ======================
# 模型評估
# ======================
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("=== Model Evaluation ===")
print(f"RMSE: {rmse:.2f}")
print(f"MAE : {mae:.2f}")
print(f"R2  : {r2:.3f}")

# ======================
#  Bootstrap 95% CI
# ======================
def bootstrap_predict(model, X_train, y_train, x_input, n_bootstrap=100):

    preds = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(len(X_train), len(X_train), replace=True)
        X_sample = X_train.iloc[idx]
        y_sample = y_train.iloc[idx]

        model.fit(X_sample, y_sample)
        pred = model.predict(x_input)[0]
        preds.append(pred)

    lower = np.percentile(preds, 2.5)
    upper = np.percentile(preds, 97.5)
    mean_pred = np.mean(preds)

    return int(mean_pred), int(lower), int(upper)

# ======================
#  預測函數（95% CI）
# ======================
def predict_event(event_dict):
    df = pd.DataFrame([event_dict])

    mean_pred, lower, upper = bootstrap_predict(
        model, X_train, y_train, df
    )

    return {
        "prediction": mean_pred,
        "CI_95_lower": lower,
        "CI_95_upper": upper
    }

# ======================
#  ROI-based Optimization
# ======================
def optimize_event(base_event, budget_weight=0.0005, n_iter=30):

    best_score = -np.inf
    best_config = None

    for _ in range(n_iter):

        scenario = base_event.copy()
        scenario['Ad_spend'] = np.random.randint(10000, 50000)
        scenario['queue_design_score'] = np.random.uniform(0.7, 1.0)
        scenario['prize_value'] = np.random.randint(5000, 20000)

        pred = predict_event(scenario)["prediction"]

        cost = scenario['Ad_spend'] + scenario['prize_value']
        score = pred - budget_weight * cost  # ROI概念

        if score > best_score:
            best_score = score
            best_config = scenario

    return best_config, best_score

# ======================
#  Feature Importance（Permutation）
# ======================
def plot_importance():
    result = permutation_importance(
        model, X_test, y_test, n_repeats=10
    )

    importance = result.importances_mean
    features = X.columns

    plt.figure()
    plt.barh(features, importance)
    plt.title("Permutation Importance")
    plt.show()

# ======================
#  模型保存
# ======================
joblib.dump(model, "event_model.pkl")

# ======================
# 測試
# ======================
if __name__ == "__main__":

    base_event = {
        'FB_exposure': 3000,
        'IG_exposure': 2500,
        'YT_exposure': 1200,
        'Ad_spend': 30000,

        'duration_hours': 6,
        'prize_value': 10000,
        'num_staff': 30,
        'queue_design_score': 0.85,

        'is_weekend': 1,
        'is_holiday': 1,
        'weather_score': 0.9,

        'past_avg': 2000,
        'past_max': 2800,
    }

    print("\n=== 單一預測（95% CI）===")
    print(predict_event(base_event))

    print("\n=== 最佳ROI方案 ===")
    best_config, score = optimize_event(base_event)
    print("最佳配置:", best_config)
    print("ROI score:", score)

    print("\n=== Feature Importance ===")
    plot_importance()