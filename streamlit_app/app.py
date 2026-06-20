import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import pairwise_distances

st.set_page_config(page_title="Voyage Analytics", layout="wide")

# ---- Load saved artifacts ----
@st.cache_resource
def load_artifacts():
    hotel_profile = joblib.load("../api/hotel_profile.pkl")
    user_profile = joblib.load("../api/user_profile.pkl")
    scaler = joblib.load("../api/recommendation_scaler.pkl")
    return hotel_profile, user_profile, scaler

hotel_profile, user_profile, scaler = load_artifacts()
hotels_scaled = scaler.transform(
    pd.DataFrame(hotel_profile[["avg_price", "avg_days_stayed"]].values, columns=["price", "days"])
)

# ---- Recommendation function (same logic as training script) ----
def recommend_hotels_for_user(user_code, top_n=5):
    user_row = user_profile[user_profile["userCode"] == user_code][
        ["avg_price_paid", "avg_trip_length"]
    ].values
    user_scaled = scaler.transform(pd.DataFrame(user_row, columns=["price", "days"]))
    distances = pairwise_distances(user_scaled, hotels_scaled, metric="euclidean")[0]
    match_scores = 1 / (1 + distances)

    result = hotel_profile.copy()
    result["match_score"] = match_scores
    return result.sort_values("match_score", ascending=False).head(top_n)


# ---- UI ----
st.title("Voyage Analytics: Travel Insights & Recommendations")
st.caption("MLOps capstone project — flight pricing, user insights, and hotel recommendations")

tab1, tab2, tab3 = st.tabs(["Hotel Recommendations", "Hotel Overview", "User Insights"])

with tab1:
    st.subheader("Get Personalized Hotel Recommendations",anchor=False)

    mode = st.radio(
        "Choose how to get recommendations:",
        ["Existing user (use booking history)", "New user (enter preferences manually)"]
    )

    if mode == "Existing user (use booking history)":
        valid_users = user_profile["userCode"].tolist()
        selected_user = st.selectbox("Select a User Code", valid_users)

        if st.button("Get Recommendations", key="existing_user_btn"):
            user_row = user_profile[user_profile["userCode"] == selected_user][
                ["avg_price_paid", "avg_trip_length"]
            ].values
            user_scaled = scaler.transform(pd.DataFrame(user_row, columns=["price", "days"]))
            distances = pairwise_distances(user_scaled, hotels_scaled, metric="euclidean")[0]
            match_scores = 1 / (1 + distances)

            recs = hotel_profile.copy()
            recs["match_score"] = match_scores
            recs = recs.sort_values("match_score", ascending=False).head(5)

            st.write(f"Top hotel matches for User {selected_user} (based on their booking history):")
            st.dataframe(
                recs[["name", "place", "avg_price", "match_score"]].style.format(
                    {"avg_price": "${:.2f}", "match_score": "{:.3f}"}
                ),
                use_container_width=True
            )
            st.bar_chart(recs.set_index("name")["match_score"])

    else:
        st.write("Tell us your travel preferences:")
        col1, col2 = st.columns(2)
        with col1:
            budget = st.slider("Preferred price per night ($)", 50, 350, 200)
        with col2:
            trip_length = st.slider("Typical trip length (days)", 1, 5, 2)

        if st.button("Get Recommendations", key="new_user_btn"):
            new_user_input = pd.DataFrame([[budget, trip_length]], columns=["price", "days"])
            user_scaled = scaler.transform(new_user_input)
            distances = pairwise_distances(user_scaled, hotels_scaled, metric="euclidean")[0]
            match_scores = 1 / (1 + distances)

            recs = hotel_profile.copy()
            recs["match_score"] = match_scores
            recs = recs.sort_values("match_score", ascending=False).head(5)

            st.write(f"Top hotel matches for budget ${budget}/night, {trip_length}-day trips:")
            st.dataframe(
                recs[["name", "place", "avg_price", "match_score"]].style.format(
                    {"avg_price": "${:.2f}", "match_score": "{:.3f}"}
                ),
                use_container_width=True
            )
            st.bar_chart(recs.set_index("name")["match_score"])

with tab2:
    st.subheader("Hotel Overview (All 9 Properties)",anchor=False)
    st.dataframe(hotel_profile, use_container_width=True)
    st.bar_chart(hotel_profile.set_index("name")["avg_price"])

with tab3:
    st.subheader("User Booking Patterns",anchor=False)
    st.write(f"Total users with booking history: {len(user_profile)}")
    st.dataframe(user_profile.describe(), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Average Price Paid Distribution")
        price_dist = user_profile["avg_price_paid"].value_counts(bins=10).sort_index()
        price_dist.index = [f"${int(interval.left)}-${int(interval.right)}" for interval in price_dist.index]
        st.bar_chart(price_dist)
    with col2:
        st.write("Average Trip Length Distribution")
        trip_dist = user_profile["avg_trip_length"].value_counts(bins=10).sort_index()
        trip_dist.index = [f"{interval.left:.1f}-{interval.right:.1f}d" for interval in trip_dist.index]
        st.bar_chart(trip_dist)