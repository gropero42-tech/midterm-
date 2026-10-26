import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

st.set_page_config(
    page_title="Campus Facility Reservation Analysis",
    page_icon="🏫",
    layout="wide"
)

st.title("🏫 Campus Facility Reservation Data Mining & Analysis")
st.caption("Data cleaning • Exploratory analysis • Reservation failure-risk prediction")

@st.cache_data
def make_dataset():
    # Reproduces the synthetic dataset from the submitted Colab notebook.
    np.random.seed(42)
    n_records = 400

    reservation_ids = [f"RES-{1000 + i}" for i in range(n_records)]
    reservation_ids[10] = reservation_ids[5]
    reservation_ids[50] = reservation_ids[45]

    user_categories = np.random.choice(
        ["Student", "Faculty", "Staff", "student", "FACULTY", None],
        size=n_records,
        p=[0.4, 0.25, 0.15, 0.1, 0.05, 0.05]
    )
    facility_types = np.random.choice(
        ["Auditorium", "Discussion Room", "Computer Lab", "Lecture Hall"],
        size=n_records
    )
    lead_times = np.random.randint(-2, 30, size=n_records)
    participants = np.random.randint(5, 100, size=n_records)
    actual_attendance = participants - np.random.randint(0, 15, size=n_records)
    actual_attendance[15] = -5
    actual_attendance[25] = 120

    usage_statuses = np.random.choice(
        ["Used", "Cancelled", "No-Show"],
        size=n_records,
        p=[0.6, 0.25, 0.15]
    )
    department_types = np.random.choice(
        ["College of CS", "College of Eng", "College of Bus", "Admin"],
        size=n_records
    )
    time_slots = np.random.choice(
        ["Morning", "Afternoon", "Evening"],
        size=n_records
    )

    raw = pd.DataFrame({
        "Reservation_ID": reservation_ids,
        "User_Category": user_categories,
        "Facility_Type": facility_types,
        "Lead_Time_Days": lead_times,
        "Participants": participants,
        "Actual_Attendance": actual_attendance,
        "Usage_Status": usage_statuses,
        "Department": department_types,
        "Time_Slot": time_slots
    })

    clean = raw.drop_duplicates(subset=["Reservation_ID"]).copy()
    clean["User_Category"] = clean["User_Category"].str.capitalize()
    clean["User_Category"] = clean["User_Category"].fillna("Unknown")
    clean["Lead_Time_Days"] = clean["Lead_Time_Days"].apply(lambda x: max(0, x))
    clean["Actual_Attendance"] = np.where(
        clean["Actual_Attendance"] < 0, 0, clean["Actual_Attendance"]
    )
    clean["Actual_Attendance"] = np.where(
        clean["Actual_Attendance"] > clean["Participants"],
        clean["Participants"],
        clean["Actual_Attendance"]
    )

    def categorize_lead_time(days):
        if days <= 2:
            return "Last Minute"
        elif days <= 7:
            return "Short Lead"
        elif days <= 14:
            return "Medium Lead"
        return "Early Booking"

    clean["Lead_Time_Category"] = clean["Lead_Time_Days"].apply(categorize_lead_time)
    clean["Reservation_Outcome"] = clean["Usage_Status"].apply(
        lambda x: "Completed" if x == "Used" else "Not Completed"
    )
    clean["Attendance_Ratio"] = np.where(
        clean["Participants"].isna() | (clean["Participants"] <= 0),
        0.0,
        clean["Actual_Attendance"] / clean["Participants"]
    )
    return raw, clean

raw_df, df = make_dataset()

predictors = [
    "User_Category", "Facility_Type", "Lead_Time_Days",
    "Department", "Time_Slot", "Participants"
]
X = pd.get_dummies(df[predictors], drop_first=True)
y = df["Reservation_Outcome"].map({"Completed": 0, "Not Completed": 1})
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Sidebar navigation
page = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Dataset Preview", "Data Cleaning", "Visualizations",
     "Predict Reservation Risk", "Model Evaluation"]
)
st.sidebar.info("This app uses the synthetic dataset generated in the original notebook.")

if page == "Dashboard":
    st.subheader("Project Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Raw records", f"{len(raw_df):,}")
    c2.metric("Clean records", f"{len(df):,}")
    c3.metric("Removed duplicates", f"{len(raw_df) - len(df):,}")
    c4.metric("Not completed", f"{(df['Reservation_Outcome'] == 'Not Completed').sum():,}")

    st.markdown("### Reservation summary")
    left, right = st.columns(2)
    with left:
        st.write("**Reservation outcome**")
        st.bar_chart(df["Reservation_Outcome"].value_counts())
    with right:
        st.write("**Bookings by facility type**")
        st.bar_chart(df["Facility_Type"].value_counts())

    failure_rate = (df["Reservation_Outcome"] == "Not Completed").mean() * 100
    st.write(f"Overall not-completed rate: **{failure_rate:.1f}%**")
    st.warning(
        "The data is synthetically generated for analysis and demonstration. "
        "Model predictions are estimates, not guaranteed outcomes."
    )

elif page == "Dataset Preview":
    st.subheader("Dataset Preview")
    view = st.radio("Choose dataset", ["Raw dataset", "Cleaned dataset"], horizontal=True)
    shown_df = raw_df if view == "Raw dataset" else df
    st.write(f"Rows: **{shown_df.shape[0]}** · Columns: **{shown_df.shape[1]}**")
    st.dataframe(shown_df, use_container_width=True, height=420)
    st.download_button(
        "Download selected dataset as CSV",
        data=shown_df.to_csv(index=False).encode("utf-8"),
        file_name="facility_reservations_raw.csv" if view == "Raw dataset" else "facility_reservations_cleaned.csv",
        mime="text/csv"
    )
    with st.expander("Summary statistics"):
        st.dataframe(shown_df.describe(include="all").transpose(), use_container_width=True)

elif page == "Data Cleaning":
    st.subheader("Data Cleaning Summary")
    duplicate_count = int(raw_df.duplicated(subset=["Reservation_ID"]).sum())
    missing_before = int(raw_df["User_Category"].isna().sum())
    negative_lead_before = int((raw_df["Lead_Time_Days"] < 0).sum())
    invalid_attendance_before = int(
        ((raw_df["Actual_Attendance"] < 0) |
         (raw_df["Actual_Attendance"] > raw_df["Participants"])).sum()
    )
    a, b, c, d = st.columns(4)
    a.metric("Duplicate IDs removed", duplicate_count)
    b.metric("Missing user categories filled", missing_before)
    c.metric("Negative lead times corrected", negative_lead_before)
    d.metric("Invalid attendance values corrected", invalid_attendance_before)

    st.markdown("### Cleaning rules applied")
    st.markdown("""
    - Removed duplicate records based on `Reservation_ID`.
    - Normalized user-category capitalization and filled missing categories with `Unknown`.
    - Changed negative lead times to zero.
    - Corrected attendance values below zero to zero and values above participant count to participant count.
    - Created `Lead_Time_Category`, `Reservation_Outcome`, and `Attendance_Ratio`.
    """)
    st.dataframe(df.head(15), use_container_width=True)

elif page == "Visualizations":
    st.subheader("Exploratory Data Analysis")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Figure 1: Reservation Count by Facility Type")
        counts = df["Facility_Type"].value_counts()
        fig, ax = plt.subplots(figsize=(8, 4.5))
        counts.plot(kind="bar", ax=ax)
        ax.set_xlabel("Facility Type")
        ax.set_ylabel("Total Reservations")
        ax.tick_params(axis="x", rotation=25)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        st.markdown("#### Figure 2: Outcome by Lead-Time Category")
        outcome_pct = pd.crosstab(
            df["Lead_Time_Category"], df["Reservation_Outcome"], normalize="index"
        ) * 100
        order = ["Last Minute", "Short Lead", "Medium Lead", "Early Booking"]
        outcome_pct = outcome_pct.reindex(order).fillna(0)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        outcome_pct.plot(kind="bar", stacked=True, ax=ax)
        ax.set_xlabel("Lead-Time Category")
        ax.set_ylabel("Percentage (%)")
        ax.tick_params(axis="x", rotation=20)
        ax.legend(title="Outcome")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("#### Not-completed rate by user category")
    failure_by_user = (
        df.groupby("User_Category")["Reservation_Outcome"]
        .apply(lambda s: (s == "Not Completed").mean() * 100)
        .sort_values(ascending=False)
    )
    st.bar_chart(failure_by_user)
    st.dataframe(failure_by_user.rename("Failure Rate (%)").round(2), use_container_width=True)

elif page == "Predict Reservation Risk":
    st.subheader("Predict Reservation Failure Risk")
    st.write("Enter details that are available when a reservation is submitted.")
    with st.form("risk_form"):
        c1, c2 = st.columns(2)
        with c1:
            user_category = st.selectbox("User category", ["Student", "Faculty", "Staff", "Unknown"])
            facility_type = st.selectbox(
                "Facility type", ["Auditorium", "Discussion Room", "Computer Lab", "Lecture Hall"]
            )
            lead_time = st.number_input("Lead time (days)", min_value=0, max_value=365, value=7, step=1)
        with c2:
            department = st.selectbox(
                "Department", ["College of CS", "College of Eng", "College of Bus", "Admin"]
            )
            time_slot = st.selectbox("Time slot", ["Morning", "Afternoon", "Evening"])
            participants = st.number_input("Expected participants", min_value=1, max_value=1000, value=30, step=1)
        submitted = st.form_submit_button("Predict risk", type="primary")

    if submitted:
        input_df = pd.DataFrame([{
            "User_Category": user_category,
            "Facility_Type": facility_type,
            "Lead_Time_Days": int(lead_time),
            "Department": department,
            "Time_Slot": time_slot,
            "Participants": int(participants)
        }])
        encoded = pd.get_dummies(input_df, drop_first=True).reindex(columns=X.columns, fill_value=0)
        pred = int(model.predict(encoded)[0])
        probabilities = model.predict_proba(encoded)[0]
        class_to_index = {int(cls): i for i, cls in enumerate(model.classes_)}
        risk_pct = float(probabilities[class_to_index[1]] * 100) if 1 in class_to_index else 0.0
        if pred == 1:
            st.error(f"Predicted result: Higher risk of non-completion ({risk_pct:.1f}% estimated risk)")
        else:
            st.success(f"Predicted result: Lower risk of non-completion ({risk_pct:.1f}% estimated risk)")
        st.caption("This is a model estimate based on synthetic training data; it should not be used as the only basis for real decisions.")

elif page == "Model Evaluation":
    st.subheader("Decision Tree Model Evaluation")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
    c2.metric("Precision", f"{precision_score(y_test, y_pred, zero_division=0):.3f}")
    c3.metric("Recall", f"{recall_score(y_test, y_pred, zero_division=0):.3f}")
    c4.metric("F1-score", f"{f1_score(y_test, y_pred, zero_division=0):.3f}")

    st.markdown("### Confusion matrix")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Completed", "Actual: Not Completed"],
        columns=["Predicted: Completed", "Predicted: Not Completed"]
    )
    st.dataframe(cm_df, use_container_width=True)

    st.markdown("### Classification report")
    report = classification_report(
        y_test, y_pred, labels=[0, 1],
        target_names=["Completed", "Not Completed"],
        output_dict=True, zero_division=0
    )
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

    st.markdown("### Top 3 influential predictors")
    importances = pd.Series(model.feature_importances_, index=X.columns).nlargest(3)
    st.bar_chart(importances)
    st.dataframe(importances.rename("Feature importance").round(4), use_container_width=True)

st.divider()
st.caption("Campus Facility Reservation & Monitoring System · Streamlit conversion of the provided notebook")
