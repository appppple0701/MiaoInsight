import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
from sklearn.inspection import permutation_importance

from xgboost import XGBRegressor

# =========================================================
# RANDOM SEED
# =========================================================

np.random.seed(42)

# =========================================================
# EVENT CATEGORY
# =========================================================

REGISTRATION_CATEGORIES = [
    "歌唱比賽",
    "書法",
    "天赦日"
]

OPEN_EVENT_CATEGORIES = [
    "乞龜",
    "關公誕辰",
    "盂蘭盆節",
    "新春團拜"
]

# =========================================================
# SYNTHETIC TEMPORAL DATA GENERATION
# =========================================================

# ---------------------------------------------------------
# REGISTRATION EVENTS
# ---------------------------------------------------------

def generate_registration_temporal_data(
    n_events=200,
    prediction_window=30
):

    rows = []

    for event_id in range(n_events):

        event_category = np.random.choice(
            REGISTRATION_CATEGORIES
        )

        venue_capacity = np.random.randint(
            500,
            10000
        )

        duration_hours = np.random.randint(
            1,
            12
        )

        historical_show_rate = np.random.uniform(
            0.55,
            0.95
        )

        category_multiplier = {

        "歌唱比賽": 0.90,
        "書法": 0.65,
        "天赦日": 0.98

        }[event_category]

        base_registration_target = np.random.randint(
            500,
            8000
        )

        cumulative_registration = 0

        previous_exposure = 0

        for days_before_event in range(
            prediction_window,
            0,
            -1
        ):

            # -------------------------------------------------
            # TIME DECAY / GROWTH
            # -------------------------------------------------

            time_progress = (
                prediction_window - days_before_event
            ) / prediction_window

            # -------------------------------------------------
            # SOCIAL EXPOSURE
            # -------------------------------------------------

            ig_exposure = int(

                np.random.randint(1000, 5000)

                + time_progress
                * np.random.randint(10000, 60000)
            )

            fb_exposure = int(

                np.random.randint(1000, 5000)

                + time_progress
                * np.random.randint(15000, 80000)
            )

            threads_exposure = int(

                np.random.randint(500, 3000)

                + time_progress
                * np.random.randint(5000, 30000)
            )

            total_exposure = (
                ig_exposure
                + fb_exposure
                + threads_exposure
            )

            # -------------------------------------------------
            # TREND VELOCITY
            # -------------------------------------------------

            trend_velocity = (
                total_exposure
                - previous_exposure
            )

            previous_exposure = total_exposure

            # -------------------------------------------------
            # REWARD SYSTEM
            # -------------------------------------------------

            reward_score = np.clip(

                np.random.normal(
                    0.4 + time_progress * 0.4,
                    0.15
                ),

                0,
                1
            )

            # -------------------------------------------------
            # MARKETING PUSH
            # -------------------------------------------------

            marketing_push_score = np.clip(

                np.random.normal(
                    0.3 + time_progress * 0.5,
                    0.2
                ),

                0,
                1
            )

            # -------------------------------------------------
            # INFLUENCER EFFECT
            # -------------------------------------------------

            influencer_score = np.clip(

                np.random.normal(
                    0.4,
                    0.2
                ),

                0,
                1
            )

            # -------------------------------------------------
            # WEATHER
            # -------------------------------------------------

            weather_score = np.clip(

                np.random.normal(
                    0.8,
                    0.1
                ),

                0.4,
                1.0
            )

            holiday_score = np.random.randint(
                0,
                2
            )

            # -------------------------------------------------
            # REGISTRATION GROWTH
            # -------------------------------------------------

            daily_registration_growth = int(

                base_registration_target
                * (
                    0.015
                    + time_progress * 0.06
                )
                * (
                    1
                    + reward_score * 0.4
                )
                * (
                    1
                    + marketing_push_score * 0.3
                )
            )

            cumulative_registration += (
                daily_registration_growth
            )

            # -------------------------------------------------
            # ATTENDANCE
            # -------------------------------------------------

            show_rate = (

                historical_show_rate

                * category_multiplier

                * weather_score

                * (
                    1
                    + holiday_score * 0.05
                )
            )

            attendance = (

                cumulative_registration
                * show_rate

                + reward_score * 600

                + influencer_score * 800

                + marketing_push_score * 500

                + trend_velocity * 0.01

                + np.random.normal(0, 150)
            )

            attendance = max(0, attendance)

            attendance = min(
                attendance,
                venue_capacity
            )

            rows.append({

                "event_id": event_id,

                "event_type":
                    "registration_based",

                "event_category":
                    event_category,

                "days_before_event":
                    days_before_event,

                "registration_count":
                    cumulative_registration,

                "registration_growth_rate":
                    daily_registration_growth,

                "ig_exposure":
                    ig_exposure,

                "fb_exposure":
                    fb_exposure,

                "threads_exposure":
                    threads_exposure,

                "trend_velocity":
                    trend_velocity,

                "reward_score":
                    reward_score,

                "marketing_push_score":
                    marketing_push_score,

                "influencer_score":
                    influencer_score,

                "historical_show_rate":
                    historical_show_rate,

                "weather_score":
                    weather_score,

                "holiday_score":
                    holiday_score,

                "duration_hours":
                    duration_hours,

                "venue_capacity":
                    venue_capacity,

                "attendance":
                    attendance
            })

    return pd.DataFrame(rows)

# ---------------------------------------------------------
# OPEN ACCESS EVENTS
# ---------------------------------------------------------

def generate_open_temporal_data(
    n_events=200,
    prediction_window=30
):

    rows = []

    for event_id in range(n_events):

        event_category = np.random.choice(
            OPEN_EVENT_CATEGORIES
        )

        venue_capacity = np.random.randint(
            1000,
            30000
        )

        category_multiplier = {

        "乞龜": 0.20,
        "關公誕辰": 0.24,
        "盂蘭盆節": 0.16,
        "新春團拜": 0.28

        }[event_category]
        base_traffic = np.random.randint(
            10000,
            100000
        )

        previous_exposure = 0

        for days_before_event in range(
            prediction_window,
            0,
            -1
        ):

            time_progress = (
                prediction_window
                - days_before_event
            ) / prediction_window

            # -------------------------------------------------
            # FOOT TRAFFIC
            # -------------------------------------------------

            nearby_foot_traffic = int(

                base_traffic

                * (
                    0.7
                    + time_progress * 0.5
                )
            )

            foot_traffic_growth = int(

                nearby_foot_traffic
                * (
                    0.02
                    + time_progress * 0.03
                )
            )

            # -------------------------------------------------
            # SOCIAL MEDIA
            # -------------------------------------------------

            ig_exposure = int(

                np.random.randint(1000, 5000)

                + time_progress
                * np.random.randint(10000, 60000)
            )

            fb_exposure = int(

                np.random.randint(1000, 5000)

                + time_progress
                * np.random.randint(10000, 80000)
            )

            threads_exposure = int(

                np.random.randint(500, 2000)

                + time_progress
                * np.random.randint(5000, 30000)
            )

            total_exposure = (
                ig_exposure
                + fb_exposure
                + threads_exposure
            )

            trend_velocity = (
                total_exposure
                - previous_exposure
            )

            previous_exposure = total_exposure

            # -------------------------------------------------
            # REWARD
            # -------------------------------------------------

            reward_score = np.clip(

                np.random.normal(
                    0.35 + time_progress * 0.45,
                    0.15
                ),

                0,
                1
            )

            # -------------------------------------------------
            # MARKETING
            # -------------------------------------------------

            marketing_push_score = np.clip(

                np.random.normal(
                    0.4 + time_progress * 0.4,
                    0.15
                ),

                0,
                1
            )

            # -------------------------------------------------
            # VENUE / WEATHER
            # -------------------------------------------------

            venue_visibility_score = np.clip(

                np.random.normal(
                    0.8,
                    0.1
                ),

                0.4,
                1.0
            )

            transport_accessibility = np.clip(

                np.random.normal(
                    0.8,
                    0.1
                ),

                0.4,
                1.0
            )

            weather_score = np.clip(

                np.random.normal(
                    0.82,
                    0.08
                ),

                0.4,
                1.0
            )

            holiday_score = np.random.randint(
                0,
                2
            )

            booth_count = np.random.randint(
                10,
                100
            )

            live_performance_score = np.clip(

                np.random.normal(
                    0.6,
                    0.2
                ),

                0,
                1
            )

            # -------------------------------------------------
            # ATTENDANCE
            # -------------------------------------------------

            attraction_rate = (

                category_multiplier

                * venue_visibility_score

                * weather_score

                * transport_accessibility

                * (
                    1
                    + holiday_score * 0.2
                )
            )

            attendance = (

                nearby_foot_traffic
                * attraction_rate

                + reward_score * 1200

                + marketing_push_score * 900

                + live_performance_score * 700

                + trend_velocity * 0.015

                + booth_count * 25

                + np.random.normal(0, 300)
            )

            attendance = max(
                0,
                attendance
            )

            attendance = min(
                attendance,
                venue_capacity
            )

            rows.append({

                "event_id": event_id,

                "event_type":
                    "open_access",

                "event_category":
                    event_category,

                "days_before_event":
                    days_before_event,

                "nearby_foot_traffic":
                    nearby_foot_traffic,

                "foot_traffic_growth":
                    foot_traffic_growth,

                "ig_exposure":
                    ig_exposure,

                "fb_exposure":
                    fb_exposure,

                "threads_exposure":
                    threads_exposure,

                "trend_velocity":
                    trend_velocity,

                "reward_score":
                    reward_score,

                "marketing_push_score":
                    marketing_push_score,

                "venue_visibility_score":
                    venue_visibility_score,

                "weather_score":
                    weather_score,

                "holiday_score":
                    holiday_score,

                "booth_count":
                    booth_count,

                "live_performance_score":
                    live_performance_score,

                "transport_accessibility":
                    transport_accessibility,

                "venue_capacity":
                    venue_capacity,

                "attendance":
                    attendance
            })

    return pd.DataFrame(rows)



# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(
    df,
    model_name="registration"
):

    y = df["attendance"]

    X = df.drop(columns=[

        "attendance",
        "event_id"
    ])

    categorical_cols = [

        "event_type",
        "event_category"
    ]

    numerical_cols = [

        col for col in X.columns
        if col not in categorical_cols
    ]

    # -----------------------------------------------------
    # PREPROCESSOR
    # -----------------------------------------------------

    preprocessor = ColumnTransformer([

        (
            "num",
            StandardScaler(),
            numerical_cols
        ),

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_cols
        )
    ])

    # -----------------------------------------------------
    # MODEL
    # -----------------------------------------------------

    model = XGBRegressor(

        objective="reg:squarederror",

        random_state=42
    )

    pipeline = Pipeline([

        ("preprocessor", preprocessor),

        ("model", model)
    ])

    # -----------------------------------------------------
    # SPLIT
    # -----------------------------------------------------

    X_train, X_test, y_train, y_test = (

        train_test_split(

            X,
            y,

            test_size=0.2,

            random_state=42
        )
    )

    # -----------------------------------------------------
    # SEARCH
    # -----------------------------------------------------

    param_dist = {

        "model__n_estimators":
            [200, 300, 500],

        "model__max_depth":
            [4, 5, 6, 8],

        "model__learning_rate":
            [0.03, 0.05, 0.1],

        "model__subsample":
            [0.8, 1.0],

        "model__colsample_bytree":
            [0.8, 1.0]
    }

    search = RandomizedSearchCV(

        pipeline,

        param_distributions=param_dist,

        n_iter=10,

        cv=3,

        scoring="neg_mean_squared_error",

        random_state=42,

        verbose=1
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # -----------------------------------------------------
    # EVALUATION
    # -----------------------------------------------------

    y_pred = best_model.predict(X_test)

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    print("\n===================================")

    print(model_name.upper())

    print("===================================")

    print(f"RMSE : {rmse:.2f}")

    print(f"MAE  : {mae:.2f}")

    print(f"R2   : {r2:.4f}")

    return (

        best_model,
        X_train,
        y_train,
        X_test,
        y_test
    )
# =========================================================
# LOAD OR TRAIN MODEL
# =========================================================

REG_MODEL_PATH = "registration_temporal_model.pkl"
OPEN_MODEL_PATH = "open_temporal_model.pkl"

REG_STD_PATH = "registration_residual_std.pkl"
OPEN_STD_PATH = "open_residual_std.pkl"

if (
    os.path.exists(REG_MODEL_PATH)
    and os.path.exists(OPEN_MODEL_PATH)
    and os.path.exists(REG_STD_PATH)
    and os.path.exists(OPEN_STD_PATH)
):

    print("\nLoading trained models...\n")

    registration_model = joblib.load(
        REG_MODEL_PATH
    )

    open_model = joblib.load(
        OPEN_MODEL_PATH
    )

    reg_residual_std = joblib.load(
        REG_STD_PATH
    )

    open_residual_std = joblib.load(
        OPEN_STD_PATH
    )

else:

    print("\nTraining new models...\n")

    registration_df = (
        generate_registration_temporal_data()
    )

    open_df = (
        generate_open_temporal_data()
    )

    (
        registration_model,
        reg_X_train,
        reg_y_train,
        reg_X_test,
        reg_y_test

    ) = train_model(
        registration_df,
        "registration"
    )

    (
        open_model,
        open_X_train,
        open_y_train,
        open_X_test,
        open_y_test

    ) = train_model(
        open_df,
        "open_access"
    )

    reg_residual_std = np.std(

        reg_y_test -

        registration_model.predict(
            reg_X_test
        )
    )

    open_residual_std = np.std(

        open_y_test -

        open_model.predict(
            open_X_test
        )
    )

    # SAVE MODEL

    joblib.dump(
        registration_model,
        REG_MODEL_PATH
    )

    joblib.dump(
        open_model,
        OPEN_MODEL_PATH
    )

    joblib.dump(
        reg_residual_std,
        REG_STD_PATH
    )

    joblib.dump(
        open_residual_std,
        OPEN_STD_PATH
    )

    print("\nModels saved.\n")

# =========================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# =========================================================

def predict_with_ci(
    model,
    x_input,
    residual_std
):

    pred = model.predict(x_input)[0]

    lower = pred - 1.96 * residual_std
    upper = pred + 1.96 * residual_std

    return {

        "prediction":
            int(pred),

        "CI_95_lower":
            int(max(0, lower)),

        "CI_95_upper":
            int(max(0, upper))
    }

# =========================================================
# EVENT PREDICTOR
# =========================================================

def predict_event(event_data):

    event_type = event_data[
        "event_type"
    ]

    df = pd.DataFrame([
        event_data
    ])

    if event_type == "registration_based":

        return predict_with_ci(

            registration_model,

            df,

            reg_residual_std
        )

    elif event_type == "open_access":

        return predict_with_ci(

            open_model,

            df,

            open_residual_std
        )

    else:

        raise ValueError(
            "Unknown event type"
        )

# =========================================================
# ATTENDANCE TIMELINE FORECAST
# =========================================================

def forecast_event_timeline(
    base_event,
    days=30
):

    predictions = []

    current_registration = 300

    for d in range(days, 0, -1):

        progress = (
            days - d
        ) / days

        event = base_event.copy()

        event["days_before_event"] = d

        # -------------------------------------------------
        # DYNAMIC GROWTH
        # -------------------------------------------------

        if event["event_type"] == \
            "registration_based":

            current_registration += int(

                120
                * (
                    1 + progress * 2
                )
            )

            event[
                "registration_count"
            ] = current_registration

            event[
                "registration_growth_rate"
            ] = int(
                current_registration * 0.05
            )

        else:

            event[
                "nearby_foot_traffic"
            ] = int(

                event[
                    "nearby_foot_traffic"
                ]
                * (
                    1 + progress * 0.3
                )
            )

        # -------------------------------------------------
        # SOCIAL GROWTH
        # -------------------------------------------------

        event["ig_exposure"] = int(

            event["ig_exposure"]
            * (
                1 + progress * 1.5
            )
        )

        event["fb_exposure"] = int(

            event["fb_exposure"]
            * (
                1 + progress * 1.2
            )
        )

        event["threads_exposure"] = int(

            event["threads_exposure"]
            * (
                1 + progress * 1.8
            )
        )

        event["trend_velocity"] = int(

            (
                event["ig_exposure"]
                + event["fb_exposure"]
                + event["threads_exposure"]
            ) * 0.03
        )

        event["reward_score"] = min(

            1.0,

            event["reward_score"]
            + progress * 0.3
        )

        event[
            "marketing_push_score"
        ] = min(

            1.0,

            event[
                "marketing_push_score"
            ]
            + progress * 0.4
        )

        result = predict_event(event)

        predictions.append({

            "days_before_event": d,

            "prediction":
                result["prediction"],

            "lower":
                result["CI_95_lower"],

            "upper":
                result["CI_95_upper"]
        })

    return pd.DataFrame(
        predictions
    )

# =========================================================
# FEATURE IMPORTANCE
# =========================================================

def plot_importance(

    model,

    X_test,

    y_test,

    title
):

    result = permutation_importance(

        model,

        X_test,

        y_test,

        n_repeats=5,

        random_state=42
    )

    feature_names = X_test.columns

    importance = (
        result.importances_mean
    )

    sorted_idx = np.argsort(
        importance
    )

    plt.figure(figsize=(10, 8))

    plt.barh(

        feature_names[sorted_idx],

        importance[sorted_idx]
    )

    plt.title(title)

    plt.xlabel("Importance")

    plt.tight_layout()

    plt.show()

# =========================================================
# PLOT TIMELINE
# =========================================================

def plot_timeline_forecast(
    forecast_df,
    title
):

    plt.figure(figsize=(12, 6))

    plt.plot(

        forecast_df[
            "days_before_event"
        ],

        forecast_df[
            "prediction"
        ],

        linewidth=3
    )

    plt.fill_between(

        forecast_df[
            "days_before_event"
        ],

        forecast_df[
            "lower"
        ],

        forecast_df[
            "upper"
        ],

        alpha=0.2
    )

    plt.gca().invert_xaxis()

    plt.title(title)

    plt.xlabel(
        "Days Before Event"
    )

    plt.ylabel(
        "Predicted Attendance"
    )

    plt.grid(True)

    plt.show()

# =========================================================
# HOURLY CROWD FLOW FORECAST
# =========================================================

def forecast_hourly_crowd_flow(
    total_attendance,
    duration_hours,
    event_category
):

    if event_category == "歌唱比賽":

        profile = np.array([
            0.20,
            0.32,
            0.22,
            0.14,
            0.08,
            0.04
        ])

    elif event_category == "書法":

        profile = np.array([
            0.30,
            0.24,
            0.18,
            0.14,
            0.09,
            0.05
        ])

    elif event_category == "天赦日":

        profile = np.array([
            0.12,
            0.18,
            0.25,
            0.22,
            0.14,
            0.09
        ])

    else:

        profile = np.array([
            0.15,
            0.20,
            0.22,
            0.18,
            0.15,
            0.10
        ])

    profile = profile[:duration_hours]

    profile = profile / profile.sum()

    hourly_people = (
        profile * total_attendance
    ).astype(int)

    rows = []

    cumulative = 0

    for hour in range(duration_hours):

        cumulative += hourly_people[hour]

        rows.append({

            "hour": hour + 1,

            "hourly_people":
                hourly_people[hour],

            "cumulative_people":
                cumulative
        })

    return pd.DataFrame(rows)

# =========================================================
# OPTIMIZATION ENGINE
# =========================================================

def optimize_event_strategy(
    event_data,
    target_attendance
):

    base_result = predict_event(event_data)

    current_prediction = base_result[
        "prediction"
    ]

    gap = (
        target_attendance
        - current_prediction
    )

    recommendations = []

    if gap <= 0:

        recommendations.append(
            "目前已達成 attendance 目標"
        )

        return {

            "current_prediction":
                current_prediction,

            "target_attendance":
                target_attendance,

            "gap":
                gap,

            "recommendations":
                recommendations
        }

    if event_data.get(
        "marketing_push_score",
        0
    ) < 0.75:

        recommendations.append(
            "增加 marketing_push_score"
        )

    if event_data.get(
        "reward_score",
        0
    ) < 0.7:

        recommendations.append(
            "增加 reward / 抽獎 / 禮品"
        )

    if event_data.get(
        "ig_exposure",
        0
    ) < 50000:

        recommendations.append(
            "提高 IG Reels / 廣告曝光"
        )

    if event_data.get(
        "fb_exposure",
        0
    ) < 50000:

        recommendations.append(
            "增加 Facebook 廣告投放"
        )

    if event_data.get(
        "threads_exposure",
        0
    ) < 20000:

        recommendations.append(
            "增加 Threads 社群互動"
        )

    if event_data.get(
        "influencer_score",
        0
    ) < 0.75:

        recommendations.append(
            "增加 KOL / influencer 合作"
        )

    if event_data.get(
        "weather_score",
        0
    ) < 0.75:

        recommendations.append(
            "需要雨備方案"
        )

    if event_data.get(
        "venue_capacity",
        0
    ) < target_attendance:

        recommendations.append(
            "場地容量可能不足"
        )

    return {

        "current_prediction":
            current_prediction,

        "target_attendance":
            target_attendance,

        "gap":
            gap,

        "recommendations":
            recommendations
    }

# =========================================================
# PLOT HOURLY FLOW
# =========================================================

def plot_hourly_flow(
    hourly_df,
    title
):

    plt.figure(figsize=(12, 6))

    plt.plot(

        hourly_df["hour"],

        hourly_df["hourly_people"],

        linewidth=3
    )

    plt.title(title)

    plt.xlabel(
        "Event Hour"
    )

    plt.ylabel(
        "People Flow"
    )

    plt.grid(True)

    plt.show()

# =========================================================
# TESTING
# =========================================================

if __name__ == "__main__":

    registration_event = {

        "event_type":
            "registration_based",

        "event_category":
            "歌唱比賽",

        "days_before_event":
            30,

        "registration_count":
            300,

        "registration_growth_rate":
            50,

        "ig_exposure":
            10000,

        "fb_exposure":
            15000,

        "threads_exposure":
            5000,

        "trend_velocity":
            2000,

        "reward_score":
            0.4,

        "marketing_push_score":
            0.5,

        "influencer_score":
            0.6,

        "historical_show_rate":
            0.88,

        "weather_score":
            0.92,

        "holiday_score":
            1,

        "duration_hours":
            6,

        "venue_capacity":
            5000
    }

    # =====================================================
    # EVENT PREDICTION
    # =====================================================

    result = predict_event(
        registration_event
    )

    print("\n=== EVENT PREDICTION ===\n")

    print(result)

    # =====================================================
    # HOURLY CROWD FLOW
    # =====================================================

    hourly_df = forecast_hourly_crowd_flow(

        total_attendance=
            result["prediction"],

        duration_hours=
            registration_event[
                "duration_hours"
            ],

        event_category=
            registration_event[
                "event_category"
            ]
    )

    print("\n=== HOURLY FLOW ===\n")

    print(hourly_df)

    plot_hourly_flow(

        hourly_df,

        "Hourly Crowd Flow"
    )

    # =====================================================
    # OPTIMIZATION
    # =====================================================

    optimization = optimize_event_strategy(

        registration_event,

        target_attendance=4500
    )

    print("\n=== OPTIMIZATION REPORT ===\n")

    print(optimization)


# =========================================================
# TESTING
# =========================================================

if __name__ == "__main__":

    # -----------------------------------------------------
    # REGISTRATION EVENT
    # -----------------------------------------------------

    registration_event = {

        "event_type":
            "registration_based",

        "event_category":
            "演唱會",

        "days_before_event":
            30,

        "registration_count":
            300,

        "registration_growth_rate":
            50,

        "ig_exposure":
            10000,

        "fb_exposure":
            15000,

        "threads_exposure":
            5000,

        "trend_velocity":
            2000,

        "reward_score":
            0.4,

        "marketing_push_score":
            0.5,

        "influencer_score":
            0.6,

        "historical_show_rate":
            0.88,

        "weather_score":
            0.92,

        "holiday_score":
            1,

        "duration_hours":
            5,

        "venue_capacity":
            5000
    }

    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    print("\n=== EVENT PREDICTION ===\n")

    result = predict_event(
        registration_event
    )

    print(result)

    # -----------------------------------------------------
    # TIMELINE FORECAST
    # -----------------------------------------------------

    forecast_df = forecast_event_timeline(

        registration_event,

        days=30
    )

    print("\n=== FORECAST TIMELINE ===\n")

    print(
        forecast_df.head()
    )

    # -----------------------------------------------------
    # PLOT TIMELINE
    # -----------------------------------------------------

    plot_timeline_forecast(

        forecast_df,

        "Attendance Forecast Timeline"
    )

    # -----------------------------------------------------
    # FEATURE IMPORTANCE
    # -----------------------------------------------------

    plot_importance(

        registration_model,

        reg_X_test,

        reg_y_test,

        "Registration Event Feature Importance"
    )