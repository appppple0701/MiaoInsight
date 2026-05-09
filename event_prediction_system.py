import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

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
    "演唱會",
    "研討會",
    "工作坊",
    "競賽"
]

OPEN_EVENT_CATEGORIES = [
    "乞龜",
    "event_1",
    "event_2"
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

            "演唱會": 0.95,
            "研討會": 0.65,
            "工作坊": 0.80,
            "競賽": 0.90

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

            "乞龜": 0.18,
            "event_1": 0.10,
            "event_2": 0.12

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
# DATASET
# =========================================================

registration_df = generate_registration_temporal_data()

open_df = generate_open_temporal_data()

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
# TRAIN MODELS
# =========================================================

registration_model, reg_X_train, reg_y_train, reg_X_test, reg_y_test = (

    train_model(
        registration_df,
        "registration"
    )
)

open_model, open_X_train, open_y_train, open_X_test, open_y_test = (

    train_model(
        open_df,
        "open_access"
    )
)

# =========================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# =========================================================

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

        X_sample = X_train.iloc[idx]

        y_sample = y_train.iloc[idx]

        model.fit(
            X_sample,
            y_sample
        )

        pred = model.predict(
            x_input
        )[0]

        preds.append(pred)

    lower = np.percentile(
        preds,
        2.5
    )

    upper = np.percentile(
        preds,
        97.5
    )

    mean_pred = np.mean(preds)

    return {

        "prediction":
            int(mean_pred),

        "CI_95_lower":
            int(lower),

        "CI_95_upper":
            int(upper)
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

        return bootstrap_predict(

            registration_model,

            reg_X_train,

            reg_y_train,

            df
        )

    elif event_type == "open_access":

        return bootstrap_predict(

            open_model,

            open_X_train,

            open_y_train,

            df
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
# SAVE MODELS
# =========================================================

joblib.dump(

    registration_model,

    "registration_temporal_model.pkl"
)

joblib.dump(

    open_model,

    "open_temporal_model.pkl"
)

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