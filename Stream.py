import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import base64

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Loan Default Risk Analytics",
    page_icon="🏦",
    layout="wide"
)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "Home"


# ============================================================
# BACKGROUND IMAGE + REDUCE TOP SPACE
# ============================================================

def set_background(image_name="Fin-Care Office.png", overlay=0.82):

    image_path = Path(__file__).parent / image_name

    if not image_path.exists():
        st.warning(f"Background image not found: {image_name}")
        return

    image_data = base64.b64encode(
        image_path.read_bytes()
    ).decode()

    st.markdown(
        f"""
        <style>
        /* Background image */
        .stApp {{
            background-image:
                linear-gradient(
                    rgba(255,255,255,{overlay}),
                    rgba(255,255,255,{overlay})
                ),
                url("data:image/png;base64,{image_data}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Reduce top space */
        .stMainBlockContainer {{
            padding-top: 0.8rem !important;
            padding-bottom: 1rem !important;
        }}

        /* Hide extra top decoration */
        [data-testid="stHeader"] {{
            background: transparent;
        }}

        /* Sidebar background */
        [data-testid="stSidebar"] {{
            background-color: rgba(245,247,251,0.96);
        }}

        /* Improve text visibility */
        h1, h2, h3, p, label {{
            color: #1f2937;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


set_background(overlay=0.82)

# ============================================================
# LOGIN PAGE
# ============================================================

if not st.session_state.logged_in:

    _, header, _ = st.columns([1, 2, 1])

    with header:

        st.markdown(
            """
            <div style="text-align:center;padding-top:20px;">
                <h1 style="font-size:19px;line-height:1.2;">
                    🏦 Loan Default Risk Analysis<br>
                    and Prediction
                </h1>
                <p style="font-size:10px;">
                    Fin-Care Starpowerz Digital Technology Pvt. Ltd.
                </p>
            """,
            unsafe_allow_html=True
        )

    st.divider()

    _, login_box, _ = st.columns([1, 1, 1])

    with login_box:

        st.subheader("🔐 Login") 

        with st.form("login_form"):

            user_id = st.text_input(
                "👤 User ID",
                placeholder="Enter User ID"
            )

            password = st.text_input(
                "🔑 Password",
                type="password",
                placeholder="Enter Password"
            )

            login = st.form_submit_button(
                "🔐 Login",
                use_container_width=True
            )

        if login:

            if user_id == "sony" and password == "Sony@4961":

                st.session_state.logged_in = True
                st.session_state.page = "Home"
                st.rerun()

            elif not user_id or not password:

                st.warning("Please enter User ID and Password.")

            else:

                st.error("Invalid User ID or Password.")

        if st.button("🔑 Forgot Password?", use_container_width=True):
            st.info("Please contact your administrator.")

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

file_path = Path(__file__).parent / "loan_default_cleaned.csv"

if not file_path.exists():
    st.error(
        "loan_default_cleaned.csv not found. "
        "Place it in the same folder as Stream.py."
    )
    st.stop()

df = pd.read_csv(file_path)

df.columns = (
    df.columns.astype(str)
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("%", "Percentage", regex=False)
)

if "Credit_Score_Cibil" in df.columns and "Credit_Score" not in df.columns:
    df.rename(columns={"Credit_Score_Cibil": "Credit_Score"}, inplace=True)


# ============================================================
# DATA CLEANING
# ============================================================

required = [
    "Loan_ID", "Loan_Type", "Age_Group", "Employment_Type",
    "Monthly_Income", "Loan_Amount", "Tenure_Months",
    "Interest_Rate", "EMI", "Credit_Score", "Previous_Default",
    "Existing_Loans", "Debt_to_Income_Ratio",
    "Loan_Approval_Date", "Outstanding_Amount",
    "Repayment_Status", "Default_Flag"
]

missing = [c for c in required if c not in df.columns]

if missing:
    st.error("Missing columns: " + ", ".join(missing))
    st.stop()

numeric_cols = [
    "Monthly_Income", "Loan_Amount", "Tenure_Months",
    "Interest_Rate", "EMI", "Credit_Score",
    "Existing_Loans", "Debt_to_Income_Ratio",
    "Outstanding_Amount", "Default_Flag"
]

for col in numeric_cols:

    df[col] = pd.to_numeric(
        df[col].astype(str).str.replace(",", "", regex=False),
        errors="coerce"
    )

df["Loan_Approval_Date"] = pd.to_datetime(
    df["Loan_Approval_Date"], errors="coerce"
)

df["Year"] = df["Loan_Approval_Date"].dt.year

df = df.dropna(subset=["Default_Flag"])

df["Default_Flag"] = df["Default_Flag"].astype(int)

df = df[df["Default_Flag"].isin([0, 1])].copy()

df["Default_Status"] = np.where(
    df["Default_Flag"] == 1,
    "Default",
    "Non-Default"
)

df["Credit_Score_Category"] = np.select(
    [
        df["Credit_Score"] < 600,
        df["Credit_Score"] < 700
    ],
    ["Poor", "Average"],
    default="Good"
)

df["DTI_Category"] = np.select(
    [
        df["Debt_to_Income_Ratio"] < 0.30,
        df["Debt_to_Income_Ratio"] < 0.50
    ],
    ["Low", "Medium"],
    default="High"
)

df["Risk_Score"] = (
    850 - df["Credit_Score"].fillna(650)
    + df["Debt_to_Income_Ratio"].fillna(0) * 100
    + df["Interest_Rate"].fillna(0) * 2
)

df["Risk_Level"] = np.select(
    [
        df["Risk_Score"] >= 250,
        df["Risk_Score"] >= 150
    ],
    ["High", "Medium"],
    default="Low"
)


# ============================================================
# MACHINE LEARNING
# ============================================================

features = [
    "Loan_Type", "Age_Group", "Employment_Type",
    "Monthly_Income", "Loan_Amount", "Tenure_Months",
    "Interest_Rate", "EMI", "Credit_Score",
    "Previous_Default", "Existing_Loans",
    "Debt_to_Income_Ratio"
]

categorical = [
    "Loan_Type", "Age_Group",
    "Employment_Type", "Previous_Default"
]

numerical = [
    "Monthly_Income", "Loan_Amount", "Tenure_Months",
    "Interest_Rate", "EMI", "Credit_Score",
    "Existing_Loans", "Debt_to_Income_Ratio"
]

X = df[features]
y = df["Default_Flag"]

if y.nunique() < 2 or y.value_counts().min() < 2:
    st.error("Dataset must contain at least two records for each default class.")
    st.stop()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

preprocessor = ColumnTransformer([
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]),
        categorical
    ),
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]),
        numerical
    )
])

logistic_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    ))
])

rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        class_weight="balanced",
        random_state=42
    ))
])

try:

    logistic_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    log_pred = logistic_model.predict(X_test)
    log_prob = logistic_model.predict_proba(X_test)[:, 1]

    rf_pred = rf_model.predict(X_test)
    rf_prob = rf_model.predict_proba(X_test)[:, 1]

except Exception as e:

    st.error(f"Model training failed: {e}")
    st.stop()


def metrics(y_true, pred, prob):

    return {
        "Accuracy": accuracy_score(y_true, pred),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1 Score": f1_score(y_true, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, prob)
    }


log_metrics = metrics(y_test, log_pred, log_prob)
rf_metrics = metrics(y_test, rf_pred, rf_prob)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏦 Loan Default Risk")

    st.caption("Fin-Care Starpowerz Digital Technology Pvt. Ltd.")

    st.divider()

    pages = [
        ("🏠 Home", "Home"),
        ("📊 Dashboard", "Dashboard"),
        ("📈 Loan Analytics", "Loan Analytics"),
        ("⚠️ Risk Analysis", "Risk Analysis"),
        ("📋 Transactions", "Transactions"),
        ("🔮 Loan Prediction", "Loan Prediction"),
        ("📉 Model Performance", "Model Performance")
    ]

    for label, page_name in pages:

        if st.button(label, use_container_width=True):
            st.session_state.page = page_name
            st.rerun()

    st.divider()

    if st.button("🚪 Logout", use_container_width=True):

        st.session_state.logged_in = False
        st.session_state.page = "Home"
        st.rerun()


page = st.session_state.page


# ============================================================
# HOME
# ============================================================

if page == "Home":

    st.title("🏦 Welcome to Loan Default Risk Analytics")

    st.write(
        "Analyze loan records, repayment patterns, credit scores "
        "and default risks using data analytics and machine learning."
    )

    st.caption("Fin-Care Starpowerz Digital Technology Pvt. Ltd.")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.container(border=True).markdown(
            "### 📊 Dashboard\nView loan KPIs and default statistics."
        )

    with c2:
        st.container(border=True).markdown(
            "### ⚠️ Risk Analysis\nExplore credit and repayment risks."
        )

    with c3:
        st.container(border=True).markdown(
            "### 🔮 Loan Prediction\nEstimate an applicant's default risk."
        )


# ============================================================
# DASHBOARD
# ============================================================

elif page == "Dashboard":

    st.title("📊 Loan Default Dashboard")

    total = len(df)
    defaults = int(df["Default_Flag"].sum())
    non_defaults = total - defaults

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Loans", f"{total:,}")
    c2.metric("Default Loans", f"{defaults:,}")
    c3.metric("Non-Default Loans", f"{non_defaults:,}")
    c4.metric("Default Rate", f"{defaults / total * 100:.2f}%")
    c5.metric("Average Loan Amount", f"₹{df['Loan_Amount'].mean():,.0f}")

    c1, c2 = st.columns(2)

    with c1:

        data = df["Default_Status"].value_counts().reset_index()
        data.columns = ["Status", "Count"]

        st.plotly_chart(
            px.pie(
                data, names="Status", values="Count",
                hole=0.4, title="Default vs Non-Default"
            ),
            use_container_width=True
        )

    with c2:

        data = df.dropna(subset=["Year"]).groupby("Year").size().reset_index(name="Loans")

        st.plotly_chart(
            px.bar(data, x="Year", y="Loans", title="Loans by Year"),
            use_container_width=True
        )

    c1, c2 = st.columns(2)

    with c1:

        data = df.groupby("Loan_Type")["Loan_Amount"].mean().reset_index()

        st.plotly_chart(
            px.bar(data, x="Loan_Type", y="Loan_Amount",
                   title="Average Loan Amount by Type"),
            use_container_width=True
        )

    with c2:

        data = df.groupby("Loan_Type")["Default_Flag"].mean().mul(100).reset_index()

        st.plotly_chart(
            px.bar(data, x="Loan_Type", y="Default_Flag",
                   title="Default Rate by Loan Type",
                   labels={"Default_Flag": "Default Rate (%)"}),
            use_container_width=True
        )


# ============================================================
# LOAN ANALYTICS
# ============================================================

elif page == "Loan Analytics":

    st.title("📈 Loan Analytics")

    c1, c2, c3 = st.columns(3)

    with c1:

        data = df["Loan_Type"].value_counts().reset_index()
        data.columns = ["Loan Type", "Count"]

        st.plotly_chart(
            px.pie(data, names="Loan Type", values="Count",
                   title="Loan Type Distribution"),
            use_container_width=True
        )

    with c2:

        data = df["Employment_Type"].value_counts().reset_index()
        data.columns = ["Employment", "Count"]

        st.plotly_chart(
            px.bar(data, x="Employment", y="Count",
                   title="Employment Distribution"),
            use_container_width=True
        )

    with c3:

        data = df["Age_Group"].value_counts().reset_index()
        data.columns = ["Age Group", "Count"]

        st.plotly_chart(
            px.bar(data, x="Age Group", y="Count",
                   title="Age Group Distribution"),
            use_container_width=True
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        data = df.groupby("Loan_Type")["Loan_Amount"].mean().reset_index()

        st.plotly_chart(
            px.bar(data, x="Loan_Type", y="Loan_Amount",
                   title="Average Loan Amount"),
            use_container_width=True
        )

    with c2:

        data = df.groupby("Loan_Type")["Credit_Score"].mean().reset_index()

        st.plotly_chart(
            px.bar(data, x="Loan_Type", y="Credit_Score",
                   title="Average Credit Score"),
            use_container_width=True
        )

    with c3:

        data = df.groupby("Loan_Type")["EMI"].mean().reset_index()

        st.plotly_chart(
            px.bar(data, x="Loan_Type", y="EMI",
                   title="Average EMI"),
            use_container_width=True
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        data = df.dropna(subset=["Year"]).groupby("Year")["Loan_Amount"].sum().reset_index()

        st.plotly_chart(
            px.line(data, x="Year", y="Loan_Amount",
                    markers=True, title="Loan Amount by Year"),
            use_container_width=True
        )

    with c2:

        data = df.dropna(subset=["Year"]).groupby("Year").size().reset_index(name="Loans")

        st.plotly_chart(
            px.bar(data, x="Year", y="Loans", title="Loan Records by Year"),
            use_container_width=True
        )

    with c3:

        data = df.groupby("Loan_Type")["Default_Flag"].mean().mul(100).reset_index()

        st.plotly_chart(
            px.bar(data, x="Loan_Type", y="Default_Flag",
                   title="Default Rate by Loan Type",
                   labels={"Default_Flag": "Default Rate (%)"}),
            use_container_width=True
        )


# ============================================================
# RISK ANALYSIS
# ============================================================

elif page == "Risk Analysis":

    st.title("⚠️ Loan Risk Analysis")

    c1, c2, c3 = st.columns(3)

    with c1:

        data = df["Risk_Level"].value_counts().reset_index()
        data.columns = ["Risk Level", "Count"]

        st.plotly_chart(
            px.pie(data, names="Risk Level", values="Count",
                   title="Risk Level Distribution"),
            use_container_width=True
        )

    with c2:

        data = df.groupby("Risk_Level")["Default_Flag"].mean().mul(100).reset_index()

        st.plotly_chart(
            px.bar(data, x="Risk_Level", y="Default_Flag",
                   title="Default Rate by Risk Level",
                   labels={"Default_Flag": "Default Rate (%)"}),
            use_container_width=True
        )

    with c3:

        data = df.groupby("Credit_Score_Category")["Default_Flag"].mean().mul(100).reset_index()

        st.plotly_chart(
            px.bar(data, x="Credit_Score_Category", y="Default_Flag",
                   title="Default Rate by Credit Score",
                   labels={"Default_Flag": "Default Rate (%)"}),
            use_container_width=True
        )

    c1, c2, c3 = st.columns(3)

    with c1:

        data = df.groupby("DTI_Category")["Default_Flag"].mean().mul(100).reset_index()

        st.plotly_chart(
            px.bar(data, x="DTI_Category", y="Default_Flag",
                   title="Default Rate by DTI",
                   labels={"Default_Flag": "Default Rate (%)"}),
            use_container_width=True
        )

    with c2:

        data = df.groupby("Previous_Default")["Default_Flag"].mean().mul(100).reset_index()

        st.plotly_chart(
            px.bar(data, x="Previous_Default", y="Default_Flag",
                   title="Previous Default Analysis",
                   labels={"Default_Flag": "Default Rate (%)"}),
            use_container_width=True
        )

    with c3:

        data = df.dropna(subset=["Year"]).groupby("Year")["Interest_Rate"].mean().reset_index()

        st.plotly_chart(
            px.line(data, x="Year", y="Interest_Rate",
                    markers=True, title="Interest Rate by Year"),
            use_container_width=True
        )

    c1, c2 = st.columns(2)

    with c1:

        st.plotly_chart(
            px.histogram(df, x="Risk_Score", nbins=25,
                         title="Risk Score Distribution"),
            use_container_width=True
        )

    with c2:

        st.plotly_chart(
            px.scatter(
                df, x="Credit_Score", y="Debt_to_Income_Ratio",
                color="Default_Status",
                hover_data=["Loan_ID", "Loan_Type", "Loan_Amount"],
                title="Credit Score vs DTI"
            ),
            use_container_width=True
        )


# ============================================================
# TRANSACTIONS
# ============================================================

elif page == "Transactions":

    st.title("📋 Loan Transactions")

    st.write(f"Total Records: {len(df):,}")

    search = st.text_input("🔍 Search Loan ID")

    display_df = df.copy()

    if search:

        display_df = display_df[
            display_df["Loan_ID"].astype(str).str.contains(
                search, case=False, na=False
            )
        ]

    columns = [
        "Loan_ID", "Loan_Type", "Age_Group", "Employment_Type",
        "Monthly_Income", "Loan_Amount", "Tenure_Months",
        "Interest_Rate", "EMI", "Credit_Score", "Previous_Default",
        "Existing_Loans", "Debt_to_Income_Ratio",
        "Loan_Approval_Date", "Outstanding_Amount",
        "Repayment_Status", "Default_Status"
    ]

    st.dataframe(
        display_df[columns],
        use_container_width=True,
        height=500
    )

    st.download_button(
        "⬇️ Download Transactions",
        display_df.to_csv(index=False).encode("utf-8"),
        file_name="loan_transactions.csv",
        mime="text/csv"
    )


# ============================================================
# LOAN PREDICTION
# ============================================================

elif page == "Loan Prediction":

    st.title("🔮 Loan Default Prediction")

    st.write("Enter applicant details to estimate default risk.")

    c1, c2, c3 = st.columns(3)

    with c1:

        loan_type = st.selectbox(
            "Loan Type",
            sorted(df["Loan_Type"].dropna().astype(str).unique())
        )

        age_group = st.selectbox(
            "Age Group",
            sorted(df["Age_Group"].dropna().astype(str).unique())
        )

        employment = st.selectbox(
            "Employment Type",
            sorted(df["Employment_Type"].dropna().astype(str).unique())
        )

        previous_default = st.selectbox(
            "Previous Default",
            sorted(df["Previous_Default"].dropna().astype(str).unique())
        )

    with c2:

        income = st.number_input(
            "Monthly Income",
            min_value=0.0,
            value=30000.0
        )

        amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=500000.0
        )

        tenure = st.number_input(
            "Tenure Months",
            min_value=1,
            value=36
        )

        interest = st.number_input(
            "Interest Rate",
            min_value=0.0,
            value=12.0
        )

    with c3:

        emi = st.number_input(
            "EMI",
            min_value=0.0,
            value=15000.0
        )

        credit = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=700
        )

        existing = st.number_input(
            "Existing Loans",
            min_value=0,
            value=1
        )

        dti = st.number_input(
            "Debt-to-Income Ratio",
            min_value=0.0,
            max_value=2.0,
            value=0.30
        )

    if st.button("🔮 Predict Loan Risk", use_container_width=True):

        input_data = pd.DataFrame([{
            "Loan_Type": loan_type,
            "Age_Group": age_group,
            "Employment_Type": employment,
            "Monthly_Income": income,
            "Loan_Amount": amount,
            "Tenure_Months": tenure,
            "Interest_Rate": interest,
            "EMI": emi,
            "Credit_Score": credit,
            "Previous_Default": previous_default,
            "Existing_Loans": existing,
            "Debt_to_Income_Ratio": dti
        }])

        prediction = logistic_model.predict(input_data)[0]

        probability = logistic_model.predict_proba(input_data)[0][1]

        st.divider()

        c1, c2 = st.columns(2)

        c1.metric(
            "Estimated Default Probability",
            f"{probability * 100:.2f}%"
        )

        c2.metric(
            "Prediction",
            "Higher Risk" if prediction == 1 else "Lower Risk"
        )

        if prediction == 1:

            st.warning(
                "⚠️ Higher loan-default risk estimated. "
                "Review the applicant carefully."
            )

        else:

            st.success(
                "✅ Lower loan-default risk estimated."
            )

        st.caption(
            "This model supports analysis and should not be used "
            "as the sole basis for lending decisions."
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title("📉 Machine Learning Model Performance")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Logistic Accuracy",
        f"{log_metrics['Accuracy'] * 100:.2f}%"
    )

    c2.metric(
        "Logistic ROC-AUC",
        f"{log_metrics['ROC-AUC'] * 100:.2f}%"
    )

    c3.metric(
        "Random Forest Accuracy",
        f"{rf_metrics['Accuracy'] * 100:.2f}%"
    )

    c4.metric(
        "Random Forest ROC-AUC",
        f"{rf_metrics['ROC-AUC'] * 100:.2f}%"
    )

    comparison = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Accuracy": [
            log_metrics["Accuracy"] * 100,
            rf_metrics["Accuracy"] * 100
        ],
        "ROC-AUC": [
            log_metrics["ROC-AUC"] * 100,
            rf_metrics["ROC-AUC"] * 100
        ]
    })

    long_data = comparison.melt(
        id_vars="Model",
        var_name="Metric",
        value_name="Score"
    )

    c1, c2 = st.columns(2)

    with c1:

        st.plotly_chart(
            px.bar(
                long_data,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                title="Model Comparison"
            ),
            use_container_width=True
        )

    with c2:

        result = pd.DataFrame({
            "Metric": list(log_metrics.keys()),
            "Logistic Regression": list(log_metrics.values()),
            "Random Forest": list(rf_metrics.values())
        })

        st.dataframe(
            result.style.format({
                "Logistic Regression": "{:.2%}",
                "Random Forest": "{:.2%}"
            }),
            use_container_width=True
        )

    c1, c2 = st.columns(2)

    with c1:

        cm = confusion_matrix(
            y_test,
            rf_pred,
            labels=[0, 1]
        )

        fig = px.imshow(
            cm,
            text_auto=True,
            x=["Non-Default", "Default"],
            y=["Non-Default", "Default"],
            labels={"x": "Predicted", "y": "Actual"},
            title="Random Forest Confusion Matrix"
        )

        st.plotly_chart(fig, use_container_width=True)

    with c2:

        fitted_preprocessor = rf_model.named_steps["preprocessor"]

        classifier = rf_model.named_steps["model"]

        feature_names = fitted_preprocessor.get_feature_names_out()

        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": classifier.feature_importances_
        })

        importance = importance.sort_values(
            "Importance", ascending=False
        ).head(12)

        importance["Feature"] = (
            importance["Feature"]
            .str.replace("cat__", "", regex=False)
            .str.replace("num__", "", regex=False)
        )

        st.plotly_chart(
            px.bar(
                importance.sort_values("Importance"),
                x="Importance",
                y="Feature",
                orientation="h",
                title="Feature Importance"
            ),
            use_container_width=True
        )

    st.caption(
        "Metrics are calculated using the held-out test dataset."
    )