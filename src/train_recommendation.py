import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances

# ---- Load data ----
users = pd.read_csv("data/users.csv")
hotels = pd.read_csv("data/hotels.csv")

# ---- Build a profile per HOTEL (since each hotel = one city) ----
hotel_profile = hotels.groupby(["name", "place"]).agg(
    avg_price=("price", "mean"),
    avg_days_stayed=("days", "mean"),
    total_bookings=("travelCode", "count")
).reset_index()

print("Hotel profiles:")
print(hotel_profile)

# ---- Build a profile per USER based on their hotel booking history ----
user_profile = hotels.groupby("userCode").agg(
    avg_price_paid=("price", "mean"),
    avg_trip_length=("days", "mean"),
    num_bookings=("travelCode", "count")
).reset_index()

print(f"\nBuilt profiles for {len(user_profile)} users")

# ---- Fit ONE scaler globally across all hotels + all users (not per-query) ----
hotel_features_raw = hotel_profile[["avg_price", "avg_days_stayed"]].values
user_features_raw = user_profile[["avg_price_paid", "avg_trip_length"]].values

scaler = StandardScaler()
scaler.fit(
    pd.DataFrame(
        list(hotel_features_raw) + list(user_features_raw),
        columns=["price", "days"]
    )
)

hotels_scaled = scaler.transform(
    pd.DataFrame(hotel_features_raw, columns=["price", "days"])
)


def recommend_hotels_for_user(user_code, top_n=5):
    if user_code not in user_profile["userCode"].values:
        return f"User {user_code} not found in booking history."

    user_row = user_profile[user_profile["userCode"] == user_code][
        ["avg_price_paid", "avg_trip_length"]
    ].values
    user_scaled = scaler.transform(
        pd.DataFrame(user_row, columns=["price", "days"])
    )

    distances = pairwise_distances(user_scaled, hotels_scaled, metric="euclidean")[0]
    match_scores = 1 / (1 + distances)

    result = hotel_profile.copy()
    result["match_score"] = match_scores
    result = result.sort_values("match_score", ascending=False).head(top_n)

    return result[["name", "place", "avg_price", "match_score"]]


# ---- Save profiles + scaler for use in the Streamlit app / API ----
joblib.dump(hotel_profile, "api/hotel_profile.pkl")
joblib.dump(user_profile, "api/user_profile.pkl")
joblib.dump(scaler, "api/recommendation_scaler.pkl")

# ---- Demo: show recommendations for a few sample users ----
for sample_user in [0, 1, 5]:
    print(f"\nTop recommendations for user {sample_user}:")
    print(recommend_hotels_for_user(sample_user))