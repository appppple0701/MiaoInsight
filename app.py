from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import os

# =====================================================
# IMPORT CORE ENGINE
# =====================================================

from event_prediction_system import (
    predict_event,
    forecast_event_timeline,
    forecast_hourly_crowd_flow,
    optimize_event_strategy
)

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
    # OPEN ACCESS EVENTS
    # =================================================

    nearby_foot_traffic: Optional[int] = None
    foot_traffic_growth: Optional[float] = None
    venue_visibility_score: Optional[float] = None
    booth_count: Optional[int] = None
    live_performance_score: Optional[float] = None
    transport_accessibility: Optional[float] = None


# =====================================================
# API - PREDICT EVENT
# =====================================================

@app.post("/predict")
def predict(data: EventInput):

    result = predict_event(data.dict())

    return result


# =====================================================
# API - TIMELINE FORECAST
# =====================================================

@app.post("/forecast_timeline")
def forecast_timeline(data: EventInput):

    result = forecast_event_timeline(
        base_event=data.dict(),
        days=data.days_before_event
    )

    return {
        "timeline": result.to_dict(orient="records")
    }


# =====================================================
# API - HOURLY CROWD FLOW
# =====================================================

@app.post("/crowd_flow")
def crowd_flow(data: EventInput):

    event_data = data.dict()

    prediction_result = predict_event(event_data)

    duration_hours = (
        event_data.get("duration_hours")
        or 6
    )

    hourly_df = forecast_hourly_crowd_flow(
        total_attendance=prediction_result["prediction"],
        duration_hours=duration_hours,
        event_category=event_data["event_category"]
    )

    return {
        "prediction": prediction_result,
        "hourly_flow": hourly_df.to_dict(orient="records")
    }


# =====================================================
# API - OPTIMIZATION
# =====================================================

@app.post("/optimize")
def optimize(data: EventInput):

    event_data = data.dict()

    target_attendance = int(
        event_data["venue_capacity"] * 0.9
    )

    result = optimize_event_strategy(
        event_data=event_data,
        target_attendance=target_attendance
    )

    return result


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
        "engine_loaded": True
    }