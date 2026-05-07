from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import pandas as pd
import numpy as np
import joblib

from sklearn.inspection import permutation_importance
from sklearn.base import clone

# ======================================
# FastAPI
# ======================================

app = FastAPI()

# ======================================
# CORS
# ======================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)

# ======================================
# load model
# ======================================

model = joblib.load("event_model.pkl")

X_train = pd.read_csv("X_train.csv")

y_train = pd.read_csv(
    "y_train.csv"
).values.ravel()

FEATURE_COLUMNS =list(X_train.columns)

# ======================================
# request schema
# ======================================

class EventInput(BaseModel):

    FB_exposure:int

    IG_exposure:int

    YT_exposure:int

    Ad_spend:int

    duration_hours:int

    prize_value:int

    num_staff:int

    queue_design_score:float

    is_weekend:int

    is_holiday:int

    weather_score:float

    past_avg:int

    past_max:int

# ======================================
# bootstrap confidence interval
# ======================================

def bootstrap_predict(

    model,

    X_train,

    y_train,

    x_input,

    n_bootstrap=30
):

    preds = []

    for _ in range(n_bootstrap):

        idx = np.random.choice(

            len(X_train),

            len(X_train),

            replace=True
        )

        X_sample =X_train.iloc[idx]

        y_sample = y_train[idx]

        new_model =clone(model)

        new_model.fit(
            X_sample,
            y_sample
        )

        pred =new_model.predict(
                x_input
            )[0]

        preds.append(pred)

    lower =np.percentile(preds,2.5)

    upper =np.percentile(preds,97.5)

    mean_pred =np.mean(preds)

    return (

        int(mean_pred),

        int(lower),

        int(upper)
    )

# ======================================
# ROI optimization
# ======================================

def optimize_event(base_event):

    original_df =pd.DataFrame([base_event])

    original_prediction =int(
            model.predict(
                original_df
            )[0]
        )

    original_cost = (

        base_event["Ad_spend"]

        +

        base_event["prize_value"]
    )

    best_score = -1e9

    best_config = None

    best_prediction = 0

    best_cost = 0

    for _ in range(100):

        scenario =base_event.copy()

        scenario["Ad_spend"] = int(

            np.random.randint(
                10000,
                50000
            )
        )

        scenario["prize_value"] = int(

            np.random.randint(
                5000,
                20000
            )
        )

        scenario["queue_design_score"] = float(

            np.random.uniform(
                0.7,
                1.0
            )
        )

        df =pd.DataFrame([scenario])

        pred =int(
                model.predict(df)[0]
            )

        cost = (

            scenario["Ad_spend"]

            +

            scenario["prize_value"]
        )

        score =pred - 0.0005 * cost

        if score > best_score:

            best_score = score

            best_config = scenario

            best_prediction = pred

            best_cost = cost

    return {

        "best_score":
            float(best_score),

        "original":{

            "prediction":
                original_prediction,

            "cost":
                int(original_cost)
        },

        "optimized":{

            "prediction":
                best_prediction,

            "cost":
                int(best_cost),

            "config":
                best_config
        }
    }

# ======================================
# cached importance
# ======================================

importance_cache = None

def get_feature_importance():

    global importance_cache

    if importance_cache is not None:

        return importance_cache

    result =permutation_importance(

            model,

            X_train,

            y_train,

            n_repeats=5,

            random_state=42
        )

    importance = {}

    for i,col in enumerate(FEATURE_COLUMNS):

        importance[col] = round(

            float(
                result.importances_mean[i]
            ),

            4
        )

    importance_cache = importance

    return importance

# ======================================
# routes
# ======================================

@app.get("/")
def home():

    return {
        "message":"AI Event Dashboard API"
    }

@app.post("/predict")
def predict(data: EventInput):

    data_dict =data.dict()

    df =pd.DataFrame([data_dict])

    # ensure feature order

    df =df[FEATURE_COLUMNS]

    # =========================
    # prediction
    # =========================

    pred =int(
            model.predict(df)[0]
        )

    # =========================
    # confidence interval
    # =========================

    mean_pred, lower, upper = (

        bootstrap_predict(

            model,

            X_train,

            y_train,

            df
        )
    )

    # =========================
    # ROI optimization
    # =========================

    roi_result =optimize_event(data_dict)

    # =========================
    # feature importance
    # =========================

    importance =get_feature_importance()

    return {

        "prediction":
            pred,

        "confidence_interval":{

            "mean":
                mean_pred,

            "lower":
                lower,

            "upper":
                upper
        },

        "roi_optimization":
            roi_result,

        "feature_importance":
            importance
    }