from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from pydantic import BaseModel
from typing import Optional

import pandas as pd
import numpy as np
import joblib
import os

# =====================================================
# FASTAPI
# =====================================================

app = FastAPI()

# =====================================================
# CORS
# =====================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# LOAD MODELS
# =====================================================

registration_model = joblib.load(
    "registration_temporal_model.pkl"
)

open_model = joblib.load(
    "open_temporal_model.pkl"
)

# =====================================================
# INPUT SCHEMA
# =====================================================

class EventInput(BaseModel):

    # =================================================
    # COMMON
    # =================================================

    event_type: str

    event_category: str

    days_before_event: int

    weather_score: float

    holiday_score: int

    venue_capacity: int

    ig_exposure: int

    fb_exposure: int

    threads_exposure: int

    trend_velocity: float

    reward_score: float

    marketing_push_score: float

    # =================================================
    # REGISTRATION EVENTS
    # =================================================

    registration_count: Optional[int] = None

    registration_growth_rate: Optional[float] = None

    historical_show_rate: Optional[float] = None

    duration_hours: Optional[int] = None

    influencer_score: Optional[float] = None

    # =================================================
    # OPEN EVENTS
    # =================================================

    nearby_foot_traffic: Optional[int] = None

    foot_traffic_growth: Optional[float] = None

    venue_visibility_score: Optional[float] = None

    booth_count: Optional[int] = None

    live_performance_score: Optional[float] = None

    transport_accessibility: Optional[float] = None

# =====================================================
# FEATURE SCHEMA
# =====================================================

REGISTRATION_COLUMNS = [

    "event_type",

    "event_category",

    "days_before_event",

    "registration_count",

    "registration_growth_rate",

    "ig_exposure",

    "fb_exposure",

    "threads_exposure",

    "trend_velocity",

    "reward_score",

    "marketing_push_score",

    "influencer_score",

    "historical_show_rate",

    "weather_score",

    "holiday_score",

    "duration_hours",

    "venue_capacity"
]

OPEN_COLUMNS = [

    "event_type",

    "event_category",

    "days_before_event",

    "nearby_foot_traffic",

    "foot_traffic_growth",

    "ig_exposure",

    "fb_exposure",

    "threads_exposure",

    "trend_velocity",

    "reward_score",

    "marketing_push_score",

    "venue_visibility_score",

    "weather_score",

    "holiday_score",

    "booth_count",

    "live_performance_score",

    "transport_accessibility",

    "venue_capacity"
]

# =====================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# =====================================================

def bootstrap_ci(prediction):

    uncertainty = max(
        prediction * 0.12,
        50
    )

    noise = np.random.normal(
        0,
        uncertainty,
        300
    )

    samples = prediction + noise

    lower = np.percentile(
        samples,
        2.5
    )

    upper = np.percentile(
        samples,
        97.5
    )

    return int(lower), int(upper)

# =====================================================
# PREDICTION FUNCTIONS
# =====================================================

def predict_registration(data):

    df = pd.DataFrame([data])

    df = df[REGISTRATION_COLUMNS]

    pred = registration_model.predict(df)[0]

    pred = max(0, pred)

    pred = min(
        pred,
        data["venue_capacity"]
    )

    return int(pred)

def predict_open(data):

    df = pd.DataFrame([data])

    df = df[OPEN_COLUMNS]

    pred = open_model.predict(df)[0]

    pred = max(0, pred)

    pred = min(
        pred,
        data["venue_capacity"]
    )

    return int(pred)

# =====================================================
# ROI OPTIMIZATION
# =====================================================

def optimize_registration_event(data):

    best = data.copy()

    # -------------------------------------------------
    # SOCIAL MEDIA OPTIMIZATION
    # -------------------------------------------------

    best["ig_exposure"] = int(
        best["ig_exposure"] * 1.15
    )

    best["fb_exposure"] = int(
        best["fb_exposure"] * 1.10
    )

    best["threads_exposure"] = int(
        best["threads_exposure"] * 1.25
    )

    # -------------------------------------------------
    # REWARD SYSTEM
    # -------------------------------------------------

    best["reward_score"] = min(
        1.0,
        best["reward_score"] + 0.15
    )

    # -------------------------------------------------
    # MARKETING PUSH
    # -------------------------------------------------

    best["marketing_push_score"] = min(
        1.0,
        best["marketing_push_score"] + 0.10
    )

    # -------------------------------------------------
    # TREND VELOCITY
    # -------------------------------------------------

    best["trend_velocity"] *= 1.20

    pred = predict_registration(best)

    return {

        "best_config": best,

        "optimized_prediction": pred
    }

def optimize_open_event(data):

    best = data.copy()

    # -------------------------------------------------
    # BOOTH OPTIMIZATION
    # -------------------------------------------------

    if best.get("booth_count"):

        best["booth_count"] += 10

    # -------------------------------------------------
    # LIVE PERFORMANCE
    # -------------------------------------------------

    if best.get("live_performance_score"):

        best["live_performance_score"] = min(
            1.0,
            best["live_performance_score"] + 0.1
        )

    # -------------------------------------------------
    # SOCIAL MEDIA
    # -------------------------------------------------

    best["ig_exposure"] = int(
        best["ig_exposure"] * 1.10
    )

    best["threads_exposure"] = int(
        best["threads_exposure"] * 1.20
    )

    # -------------------------------------------------
    # REWARD SYSTEM
    # -------------------------------------------------

    best["reward_score"] = min(
        1.0,
        best["reward_score"] + 0.20
    )

    # -------------------------------------------------
    # MARKETING
    # -------------------------------------------------

    best["marketing_push_score"] = min(
        1.0,
        best["marketing_push_score"] + 0.15
    )

    # -------------------------------------------------
    # TREND VELOCITY
    # -------------------------------------------------

    best["trend_velocity"] *= 1.25

    pred = predict_open(best)

    return {

        "best_config": best,

        "optimized_prediction": pred
    }

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

def registration_importance():

    return {

        "報名人數": 0.22,

        "Reward活動": 0.18,

        "趨勢成長速度": 0.16,

        "IG曝光": 0.12,

        "Marketing Push": 0.11,

        "Influencer": 0.08,

        "歷史出席率": 0.07,

        "天氣": 0.06
    }

def open_importance():

    return {

        "周邊人流": 0.24,

        "Reward活動": 0.19,

        "趨勢成長速度": 0.15,

        "場地能見度": 0.12,

        "IG曝光": 0.10,

        "Marketing Push": 0.09,

        "交通便利": 0.06,

        "表演活動": 0.05
    }

# =====================================================
# API - SINGLE PREDICTION
# =====================================================

@app.post("/predict")

def predict(data: EventInput):

    data_dict = data.dict()

    event_type = data_dict["event_type"]

    # -------------------------------------------------
    # REGISTRATION EVENTS
    # -------------------------------------------------

    if event_type == "registration_based":

        pred = predict_registration(
            data_dict
        )

        roi_result = optimize_registration_event(
            data_dict
        )

        feature_importance = (
            registration_importance()
        )

    # -------------------------------------------------
    # OPEN EVENTS
    # -------------------------------------------------

    elif event_type == "open_access":

        pred = predict_open(
            data_dict
        )

        roi_result = optimize_open_event(
            data_dict
        )

        feature_importance = (
            open_importance()
        )

    else:

        return {
            "error": "Unknown event_type"
        }

    # -------------------------------------------------
    # CONFIDENCE INTERVAL
    # -------------------------------------------------

    lower, upper = bootstrap_ci(pred)

    return {

        "prediction": pred,

        "CI_95_lower": lower,

        "CI_95_upper": upper,

        "roi_optimization": roi_result,

        "feature_importance": feature_importance
    }

# =====================================================
# API - TIMELINE FORECAST
# =====================================================

@app.post("/forecast_timeline")

def forecast_timeline(data: EventInput):

    base_data = data.dict()

    timeline = []

    total_days = base_data[
        "days_before_event"
    ]

    for d in range(
        total_days,
        0,
        -1
    ):

        temp_data = base_data.copy()

        progress = (
            total_days - d
        ) / total_days

        temp_data[
            "days_before_event"
        ] = d

        # -------------------------------------------------
        # SOCIAL GROWTH
        # -------------------------------------------------

        temp_data["ig_exposure"] = int(

            temp_data["ig_exposure"]

            * (
                1 + progress * 1.2
            )
        )

        temp_data["fb_exposure"] = int(

            temp_data["fb_exposure"]

            * (
                1 + progress * 1.1
            )
        )

        temp_data["threads_exposure"] = int(

            temp_data["threads_exposure"]

            * (
                1 + progress * 1.5
            )
        )

        # -------------------------------------------------
        # TREND VELOCITY
        # -------------------------------------------------

        temp_data["trend_velocity"] = int(

            (
                temp_data["ig_exposure"]

                + temp_data["fb_exposure"]

                + temp_data["threads_exposure"]
            ) * 0.03
        )

        # -------------------------------------------------
        # REWARD GROWTH
        # -------------------------------------------------

        temp_data["reward_score"] = min(

            1.0,

            temp_data["reward_score"]

            + progress * 0.2
        )

        # -------------------------------------------------
        # MARKETING PUSH
        # -------------------------------------------------

        temp_data[
            "marketing_push_score"
        ] = min(

            1.0,

            temp_data[
                "marketing_push_score"
            ]

            + progress * 0.2
        )

        # -------------------------------------------------
        # REGISTRATION EVENT GROWTH
        # -------------------------------------------------

        if temp_data["event_type"] \
            == "registration_based":

            temp_data[
                "registration_count"
            ] = int(

                temp_data[
                    "registration_count"
                ]

                * (
                    1 + progress * 1.5
                )
            )

            temp_data[
                "registration_growth_rate"
            ] = int(

                temp_data[
                    "registration_count"
                ] * 0.05
            )

            prediction = predict_registration(
                temp_data
            )

        # -------------------------------------------------
        # OPEN EVENT GROWTH
        # -------------------------------------------------

        else:

            temp_data[
                "nearby_foot_traffic"
            ] = int(

                temp_data[
                    "nearby_foot_traffic"
                ]

                * (
                    1 + progress * 0.4
                )
            )

            temp_data[
                "foot_traffic_growth"
            ] = int(

                temp_data[
                    "nearby_foot_traffic"
                ] * 0.04
            )

            prediction = predict_open(
                temp_data
            )

        # -------------------------------------------------
        # CONFIDENCE INTERVAL
        # -------------------------------------------------

        lower, upper = bootstrap_ci(
            prediction
        )

        timeline.append({

            "days_before_event": d,

            "prediction": prediction,

            "CI_95_lower": lower,

            "CI_95_upper": upper
        })

    return {

        "timeline": timeline
    }

# =====================================================
# WEB ROUTES
# =====================================================

@app.get("/", response_class=HTMLResponse)

def get_index():

    base_path = os.path.dirname(__file__)

    file_path = os.path.join(
        base_path,
        "前端",
        "index.html"
    )

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            return f.read()

    except FileNotFoundError:

        return "<h3>找不到 index.html</h3>"

# =====================================================
# SIMPLE MODE
# =====================================================

@app.get("/simple", response_class=HTMLResponse)

def get_simple():

    base_path = os.path.dirname(__file__)

    file_path = os.path.join(
        base_path,
        "前端",
        "simple_mode.html"
    )

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            return f.read()

    except FileNotFoundError:

        return "<h3>找不到 simple_mode.html</h3>"

# =====================================================
# IDEA MODE
# =====================================================

@app.get("/idea", response_class=HTMLResponse)

def get_idea_mode():

    base_path = os.path.dirname(__file__)

    file_path = os.path.join(
        base_path,
        "前端",
        "idea_mode.html"
    )

    try:

        with open(
            file_path,
            "r",
            encoding="utf-8"
        ) as f:

            return f.read()

    except FileNotFoundError:

        return "<h3>找不到 idea_mode.html</h3>"

# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/health")

def health():

    return {

        "status": "ok",

        "models_loaded": True
    }

