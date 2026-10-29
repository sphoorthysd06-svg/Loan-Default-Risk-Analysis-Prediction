import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
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

# Consistent professional chart styling across every page.
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#356AE6", "#14A38B", "#F59E0B", "#8B5CF6", "#64748B"]
px.defaults.width = None
px.defaults.height = 360

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
    st.session_state.page = "Dashboard"


# ============================================================
# BACKGROUND IMAGE + REDUCE TOP SPACE
# ============================================================

def set_background(image_name="Fin-Care Office.png"):

    # The image is used only while the login page is displayed.
    image_path = Path(__file__).parent / image_name

    if not image_path.exists():
        st.warning(
            f"Background image not found: {image_name}. "
            "Place it in the same folder as Stream.py."
        )
        return

    image_data = base64.b64encode(image_path.read_bytes()).decode()

    # Match the friend's background style, with a readable translucent login panel.
    image_mime = "image/jpeg" if image_path.suffix.lower() in [".jpg", ".jpeg"] else "image/png"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:{image_mime};base64,{image_data}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        .main {{
            background: transparent !important;
        }}

        .block-container {{
            background: rgba(255, 255, 255, 0.88);
            border-radius: 15px;
            padding: 2rem;
        }}

        .stMainBlockContainer {{
            padding-top: 0.8rem !important;
            padding-bottom: 1rem !important;
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# LOGIN PAGE
# ============================================================

if not st.session_state.logged_in:

    # Apply the background only on the login screen.
    set_background()

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
                st.session_state.page = "Dashboard"
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

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1400px; }
    [data-testid="stSidebar"] { background-color: #F4F7FB; }
    [data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E2E8F0; padding: 14px; border-radius: 10px; }
    h1, h2, h3 { color: #183153; }
    div[data-testid="stPlotlyChart"] { background: #FFFFFF; border: 1px solid #E8EDF4; border-radius: 10px; padding: 8px; }
    </style>
    """,
    unsafe_allow_html=True
)

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

# DEMONSTRATION DATA LABELS
# Keep all source records so the dashboard KPIs remain consistent with the
# full 2,000-row dataset. The synthetic dataset contains age bands rather than
# exact ages, so bands above the requested 55-year limit are displayed as
# "51-55 years" for demonstration purposes (not as verified individual ages).
if "Age_Group" in df.columns:
    df["Age_Group"] = df["Age_Group"].astype(str).str.strip()
    age_label_map = {
        "21-30": "21-30 years", "21–30": "21-30 years", "21-30 years": "21-30 years",
        "31-40": "31-40 years", "31–40": "31-40 years", "31-40 years": "31-40 years",
        "41-50": "41-50 years", "41–50": "41-50 years", "41-50 years": "41-50 years",
        "51-60": "51-55 years", "51–60": "51-55 years", "51-60 years": "51-55 years",
        "56-60": "51-55 years", "56-60 years": "51-55 years",
        "60+": "51-55 years", "60+ years": "51-55 years",
        "61+": "51-55 years", "61+ years": "51-55 years",
        "Above 60": "51-55 years", "Above 60 years": "51-55 years",
    }
    df["Age_Group"] = df["Age_Group"].replace(age_label_map)
if "Employment_Type" in df.columns:
    df["Employment_Type"] = df["Employment_Type"].astype(str).str.strip()
    df["Employment_Type"] = df["Employment_Type"].replace({
        "Business Owner": "Self-Employed",
        "Contract": "Self-Employed",
    })


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
        df["Credit_Score"] < 650,
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
    
    st.caption(
    "📍 Anajneya Complex, Lingad Gudi Road, "
    "Vijayapura, Karnataka – 586101"
)
    
    st.divider()

    pages = [
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
        st.session_state.page = "Dashboard"
        st.rerun()


page = st.session_state.page

# ============================================================
# DASHBOARD — CLEAN FULL-WIDTH LAYOUT
# ============================================================

if page == "Dashboard":
    st.title("📊 Loan Default Dashboard")
    st.caption("Portfolio overview, loan trends and default performance.")
    st.divider()

    total = len(df)
    defaults = int(df["Default_Flag"].sum())
    non_defaults = total - defaults
    metrics = st.columns(5)
    metrics[0].metric("Total Loans", f"{total:,}")
    metrics[1].metric("Default Loans", f"{defaults:,}")
    metrics[2].metric("Non-Default Loans", f"{non_defaults:,}")
    metrics[3].metric("Default Rate", f"{defaults / total * 100:.2f}%" if total else "0.00%")
    metrics[4].metric("Average Loan Amount", f"₹{df['Loan_Amount'].mean():,.0f}")

    # Karnataka locations and bank/financial institution reference
    # Only this location/map section has been updated.
    st.subheader("Karnataka Loan Service Locations")
    st.caption(
        "Choose a city or click a red map marker to view example banks and open their locations in Google Maps. "
        "The markers represent city centres, not verified bank-branch coordinates."
    )

    # Google Maps embed: no Folium or streamlit-folium required.
    import urllib.parse
    import streamlit.components.v1 as components

    karnataka_locations = [
        {"City": "Bengaluru", "Latitude": 12.9716, "Longitude": 77.5946, "Institutions": ["State Bank of India (SBI)", "HDFC Bank", "ICICI Bank", "Axis Bank", "Canara Bank"]},
        {"City": "Mysuru", "Latitude": 12.2958, "Longitude": 76.6394, "Institutions": ["State Bank of India (SBI)", "Bank of Baroda", "HDFC Bank", "ICICI Bank"]},
        {"City": "Hubballi", "Latitude": 15.3647, "Longitude": 75.1240, "Institutions": ["State Bank of India (SBI)", "Canara Bank", "HDFC Bank", "Axis Bank"]},
        {"City": "Belagavi", "Latitude": 15.8497, "Longitude": 74.4977, "Institutions": ["State Bank of India (SBI)", "Bank of Baroda", "ICICI Bank"]},
        {"City": "Vijayapura", "Latitude": 16.8302, "Longitude": 75.7100, "Institutions": ["State Bank of India (SBI)", "Bank of Baroda", "HDFC Bank", "ICICI Bank", "Axis Bank", "Union Bank of India"]},
        {"City": "Kalaburagi", "Latitude": 17.3297, "Longitude": 76.8343, "Institutions": ["State Bank of India (SBI)", "Canara Bank", "Bank of Baroda"]},
        {"City": "Mangaluru", "Latitude": 12.9141, "Longitude": 74.8560, "Institutions": ["State Bank of India (SBI)", "Canara Bank", "Karnataka Bank", "HDFC Bank"]},
        {"City": "Davanagere", "Latitude": 14.4644, "Longitude": 75.9218, "Institutions": ["State Bank of India (SBI)", "Canara Bank", "Bank of Baroda"]},
        {"City": "Shivamogga", "Latitude": 13.9299, "Longitude": 75.5681, "Institutions": ["State Bank of India (SBI)", "Canara Bank", "HDFC Bank"]},
        {"City": "Tumakuru", "Latitude": 13.3379, "Longitude": 77.1173, "Institutions": ["State Bank of India (SBI)", "Canara Bank", "ICICI Bank"]},
        {"City": "Udupi", "Latitude": 13.3409, "Longitude": 74.7421, "Institutions": ["State Bank of India (SBI)", "Karnataka Bank", "Canara Bank"]},
        {"City": "Ballari", "Latitude": 15.1394, "Longitude": 76.9214, "Institutions": ["State Bank of India (SBI)", "Canara Bank", "Bank of Baroda"]},
        {"City": "Raichur", "Latitude": 16.2120, "Longitude": 77.3439, "Institutions": ["State Bank of India (SBI)", "Canara Bank", "Union Bank of India"]},
        {"City": "Hassan", "Latitude": 13.0068, "Longitude": 76.1003, "Institutions": ["State Bank of India (SBI)", "Canara Bank", "HDFC Bank"]},
        {"City": "Kolar", "Latitude": 13.1367, "Longitude": 78.1292, "Institutions": ["State Bank of India (SBI)", "Canara Bank", "ICICI Bank"]},
        {"City": "Chikkamagaluru", "Latitude": 13.3161, "Longitude": 75.7720, "Institutions": ["State Bank of India (SBI)", "Canara Bank", "Karnataka Bank"]},
    ]
    locations_df = pd.DataFrame(karnataka_locations)
    city_names = locations_df["City"].tolist()

    if "selected_map_city" not in st.session_state or st.session_state.selected_map_city not in city_names:
        st.session_state.selected_map_city = "Vijayapura"

    map_col, details_col = st.columns([1.5, 1])
    with map_col:
        st.markdown("**Choose a Karnataka city to view it on Google Maps**")
        selected_city = st.selectbox(
            "Karnataka city",
            city_names,
            index=city_names.index(st.session_state.selected_map_city),
            key="karnataka_city_selector",
        )
        st.session_state.selected_map_city = selected_city
        selected_location = locations_df.loc[locations_df["City"] == selected_city].iloc[0]

        map_query = urllib.parse.quote(f"{selected_city}, Karnataka, India")
        google_map_url = f"https://maps.google.com/maps?q={map_query}&z=10&output=embed"
        components.iframe(google_map_url, height=440, scrolling=False)
        st.caption("This is an embedded Google Map for the selected city. Choose a bank on the right to search for its branches.")

    with details_col:
        st.markdown("**Selected location details**")
        st.markdown(f"**City:** {selected_location['City']}, Karnataka")
        st.markdown("**Banks to explore in this city:**")
        for institution in selected_location["Institutions"]:
            st.write(f"• {institution}")
            query = f"{institution} {selected_location['City']} Karnataka".replace(" ", "+")
            st.link_button(
                f"📍 Find {institution} in Google Maps",
                f"https://www.google.com/maps/search/?api=1&query={query}",
                use_container_width=True,
                key=f"map_{selected_location['City']}_{institution}".replace(" ", "_").replace("(", "").replace(")", ""),
            )

    st.markdown("**Office location**")
    st.write("Fin-Care Starpowerz Digital Technology Pvt. Ltd.")
    st.write("Anjaneya Complex, Lingad Gudi Road, Vijayapura, Karnataka – 586101")
    st.link_button(
        "📍 Open office location in Google Maps",
        "https://www.google.com/maps/search/?api=1&query=Anjaneya+Complex+Lingad+Gudi+Road+Vijayapura+Karnataka+586101",
        use_container_width=True,
    )

    st.divider()
    st.subheader("Default Portfolio")
    data = df["Default_Status"].value_counts().rename_axis("Status").reset_index(name="Count")
    fig = px.pie(data, names="Status", values="Count", hole=0.48, title="Default vs Non-Default")
    fig.update_traces(textposition="inside", textinfo="percent+label")
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=25, r=25, t=70, b=30))
    # Keep charts compact, with slimmer bars and consistent typography.
    fig.update_layout(
        height=320,
        font=dict(size=11),
        title_font=dict(size=14),
        margin=dict(l=35, r=25, t=65, b=55),
    )
    fig.update_traces(width=0.32, selector=dict(type="bar"))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Loan Volume Over Time")
    data = df.dropna(subset=["Year"]).groupby("Year").size().reset_index(name="Loans")
    fig = px.bar(data, x="Year", y="Loans", title="Loan Records by Year", text_auto=True)
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=25, r=25, t=70, b=35),
                      xaxis_title="Year", yaxis_title="Number of loans")
    # Keep charts compact, with slimmer bars and consistent typography.
    fig.update_layout(
        height=320,
        font=dict(size=11),
        title_font=dict(size=14),
        margin=dict(l=35, r=25, t=65, b=55),
    )
    fig.update_traces(width=0.32, selector=dict(type="bar"))
    st.plotly_chart(fig, use_container_width=True)

    data = df.groupby("Loan_Type", as_index=False)["Loan_Amount"].mean()
    fig = px.bar(data, x="Loan_Type", y="Loan_Amount", title="Average Loan Amount by Loan Type", text_auto=".2s")
    fig.update_layout(template="plotly_white", height=440, margin=dict(l=25, r=25, t=70, b=90),
                      xaxis_title="Loan type", yaxis_title="Average loan amount", xaxis_tickangle=-15)
    # Keep charts compact, with slimmer bars and consistent typography.
    fig.update_layout(
        height=320,
        font=dict(size=11),
        title_font=dict(size=14),
        margin=dict(l=35, r=25, t=65, b=55),
    )
    fig.update_traces(width=0.32, selector=dict(type="bar"))
    st.plotly_chart(fig, use_container_width=True)

    data = df.groupby("Loan_Type", as_index=False)["Default_Flag"].mean()
    data["Default_Rate"] = data["Default_Flag"] * 100
    fig = px.bar(data, x="Loan_Type", y="Default_Rate", title="Default Rate by Loan Type", text_auto=".1f")
    fig.update_layout(template="plotly_white", height=440, margin=dict(l=25, r=25, t=70, b=90),
                      xaxis_title="Loan type", yaxis_title="Default rate (%)", xaxis_tickangle=-15)
    # Keep charts compact, with slimmer bars and consistent typography.
    fig.update_layout(
        height=320,
        font=dict(size=11),
        title_font=dict(size=14),
        margin=dict(l=35, r=25, t=65, b=55),
    )
    fig.update_traces(width=0.32, selector=dict(type="bar"))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# LOAN ANALYTICS — ONE FULL-WIDTH CHART PER ROW
# ============================================================

elif page == "Loan Analytics":
    st.title("📈 Loan Analytics")
    st.caption("Explore borrower profiles, loan values and yearly portfolio trends.")
    st.divider()

    data = df["Loan_Type"].value_counts().rename_axis("Loan Type").reset_index(name="Count")
    fig = px.bar(data, x="Loan Type", y="Count", title="Loan Type Distribution", text_auto=True)
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=25, r=25, t=70, b=85),
                      xaxis_tickangle=-15, xaxis_title="Loan type", yaxis_title="Number of loans")
    # Keep charts compact, with slimmer bars and consistent typography.
    fig.update_layout(
        height=320,
        font=dict(size=11),
        title_font=dict(size=14),
        margin=dict(l=35, r=25, t=65, b=55),
    )
    fig.update_traces(width=0.32, selector=dict(type="bar"))
    st.plotly_chart(fig, use_container_width=True)

    data = df["Employment_Type"].value_counts().rename_axis("Employment").reset_index(name="Count")
    fig = px.bar(data, x="Employment", y="Count", title="Employment Distribution", text_auto=True)
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=25, r=25, t=70, b=65),
                      xaxis_title="Employment type", yaxis_title="Number of borrowers")
    # Keep charts compact, with slimmer bars and consistent typography.
    fig.update_layout(
        height=320,
        font=dict(size=11),
        title_font=dict(size=14),
        margin=dict(l=35, r=25, t=65, b=55),
    )
    fig.update_traces(width=0.32, selector=dict(type="bar"))
    st.plotly_chart(fig, use_container_width=True)

    data = df["Age_Group"].value_counts().rename_axis("Age Group").reset_index(name="Count")
    fig = px.bar(data, x="Age Group", y="Count", title="Borrower Age Group Distribution", text_auto=True)
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=25, r=25, t=70, b=55),
                      xaxis_title="Age group", yaxis_title="Number of borrowers")
    # Keep charts compact, with slimmer bars and consistent typography.
    fig.update_layout(
        height=320,
        font=dict(size=11),
        title_font=dict(size=14),
        margin=dict(l=35, r=25, t=65, b=55),
    )
    fig.update_traces(width=0.32, selector=dict(type="bar"))
    st.plotly_chart(fig, use_container_width=True)

    for column, title, y_label, fmt in [
        ("Loan_Amount", "Average Loan Amount by Loan Type", "Average loan amount", ".2s"),
        ("Credit_Score", "Average Credit Score by Loan Type", "Average credit score", ".1f"),
        ("EMI", "Average EMI by Loan Type", "Average EMI", ".2s"),
    ]:
        data = df.groupby("Loan_Type", as_index=False)[column].mean()
        fig = px.bar(data, x="Loan_Type", y=column, title=title, text_auto=fmt)
        fig.update_layout(template="plotly_white", height=420, margin=dict(l=25, r=25, t=70, b=85),
                          xaxis_title="Loan type", yaxis_title=y_label, xaxis_tickangle=-15)
        # Keep charts compact, with slimmer bars and consistent typography.
    fig.update_layout(
        height=320,
        font=dict(size=11),
        title_font=dict(size=14),
        margin=dict(l=35, r=25, t=65, b=55),
    )
    fig.update_traces(width=0.32, selector=dict(type="bar"))
    st.plotly_chart(fig, use_container_width=True)

    data = df.dropna(subset=["Year"]).groupby("Year", as_index=False)["Loan_Amount"].sum()
    fig = px.line(data, x="Year", y="Loan_Amount", markers=True, title="Total Loan Amount by Year")
    fig.update_traces(line_width=3, marker_size=8)
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=25, r=25, t=70, b=35),
                      xaxis_title="Year", yaxis_title="Total loan amount")
    # Keep charts compact, with slimmer bars and consistent typography.
    fig.update_layout(
        height=320,
        font=dict(size=11),
        title_font=dict(size=14),
        margin=dict(l=35, r=25, t=65, b=55),
    )
    fig.update_traces(width=0.32, selector=dict(type="bar"))
    st.plotly_chart(fig, use_container_width=True)

    data = df.dropna(subset=["Year"]).groupby("Year").size().reset_index(name="Loans")
    fig = px.bar(data, x="Year", y="Loans", title="Loan Records by Year", text_auto=True)
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=25, r=25, t=70, b=35),
                      xaxis_title="Year", yaxis_title="Number of loans")
    # Keep charts compact, with slimmer bars and consistent typography.
    fig.update_layout(
        height=320,
        font=dict(size=11),
        title_font=dict(size=14),
        margin=dict(l=35, r=25, t=65, b=55),
    )
    fig.update_traces(width=0.32, selector=dict(type="bar"))
    st.plotly_chart(fig, use_container_width=True)

    data = df.groupby("Loan_Type", as_index=False)["Default_Flag"].mean()
    data["Default_Rate"] = data["Default_Flag"] * 100
    fig = px.bar(data, x="Loan_Type", y="Default_Rate", title="Default Rate by Loan Type", text_auto=".1f")
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=25, r=25, t=70, b=85),
                      xaxis_title="Loan type", yaxis_title="Default rate (%)", xaxis_tickangle=-15)
    # Keep charts compact, with slimmer bars and consistent typography.
    fig.update_layout(
        height=320,
        font=dict(size=11),
        title_font=dict(size=14),
        margin=dict(l=35, r=25, t=65, b=55),
    )
    fig.update_traces(width=0.32, selector=dict(type="bar"))
    st.plotly_chart(fig, use_container_width=True)


# ============================================================
# RISK ANALYSIS — ONE FULL-WIDTH CHART PER ROW
# ============================================================

elif page == "Risk Analysis":
    st.title("⚠️ Loan Risk Analysis")
    st.caption("Review risk categories and borrower factors associated with loan defaults.")
    st.divider()

    data = df["Risk_Level"].value_counts().rename_axis("Risk Level").reset_index(name="Count")
    order = [x for x in ["Low", "Medium", "High"] if x in data["Risk Level"].tolist()]
    fig = px.bar(data, x="Risk Level", y="Count", title="Risk Level Distribution",
                 category_orders={"Risk Level": order}, text_auto=True)
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=25, r=25, t=70, b=35),
                      xaxis_title="Risk level", yaxis_title="Number of loans")
    # Keep charts compact, with slimmer bars and consistent typography.
    fig.update_layout(
        height=320,
        font=dict(size=11),
        title_font=dict(size=14),
        margin=dict(l=35, r=25, t=65, b=55),
    )
    fig.update_traces(width=0.32, selector=dict(type="bar"))
    st.plotly_chart(fig, use_container_width=True)

    data = df.groupby("Risk_Level", as_index=False)["Default_Flag"].mean()
    data["Default_Rate"] = data["Default_Flag"] * 100
    fig = px.bar(data, x="Risk_Level", y="Default_Rate", title="Default Rate by Risk Level",
                 category_orders={"Risk_Level": order}, text_auto=".1f")
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=25, r=25, t=70, b=35),
                      xaxis_title="Risk level", yaxis_title="Default rate (%)")
    # Keep charts compact, with slimmer bars and consistent typography.
    fig.update_layout(
        height=320,
        font=dict(size=11),
        title_font=dict(size=14),
        margin=dict(l=35, r=25, t=65, b=55),
    )
    fig.update_traces(width=0.32, selector=dict(type="bar"))
    st.plotly_chart(fig, use_container_width=True)

    data = df.groupby("Credit_Score_Category", as_index=False)["Default_Flag"].mean()
    data["Default_Rate"] = data["Default_Flag"] * 100
    fig = px.bar(data, x="Credit_Score_Category", y="Default_Rate",
                 title="Default Rate by Credit Score Category", text_auto=".1f")
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=25, r=25, t=70, b=35),
                      xaxis_title="Credit score category", yaxis_title="Default rate (%)")
    # Keep charts compact, with slimmer bars and consistent typography.
    fig.update_layout(
        height=320,
        font=dict(size=11),
        title_font=dict(size=14),
        margin=dict(l=35, r=25, t=65, b=55),
    )
    fig.update_traces(width=0.32, selector=dict(type="bar"))
    st.plotly_chart(fig, use_container_width=True)

    data = df.groupby("DTI_Category", as_index=False)["Default_Flag"].mean()
    data["Default_Rate"] = data["Default_Flag"] * 100
    fig = px.bar(data, x="DTI_Category", y="Default_Rate", title="Default Rate by Debt-to-Income Category",
                 text_auto=".1f", category_orders={"DTI_Category": ["Low", "Medium", "High"]})
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=25, r=25, t=70, b=35),
                      xaxis_title="Debt-to-income category", yaxis_title="Default rate (%)")
    # Keep charts compact, with slimmer bars and consistent typography.
    fig.update_layout(
        height=320,
        font=dict(size=11),
        title_font=dict(size=14),
        margin=dict(l=35, r=25, t=65, b=55),
    )
    fig.update_traces(width=0.32, selector=dict(type="bar"))
    st.plotly_chart(fig, use_container_width=True)

    data = df.groupby("Previous_Default", as_index=False)["Default_Flag"].mean()
    data["Default_Rate"] = data["Default_Flag"] * 100
    fig = px.bar(data, x="Previous_Default", y="Default_Rate",
                 title="Current Default Rate by Previous Default History", text_auto=".1f")
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=25, r=25, t=70, b=35),
                      xaxis_title="Previous default history", yaxis_title="Default rate (%)")
    # Keep charts compact, with slimmer bars and consistent typography.
    fig.update_layout(
        height=320,
        font=dict(size=11),
        title_font=dict(size=14),
        margin=dict(l=35, r=25, t=65, b=55),
    )
    fig.update_traces(width=0.32, selector=dict(type="bar"))
    st.plotly_chart(fig, use_container_width=True)

    data = df.dropna(subset=["Year"]).groupby("Year", as_index=False)["Interest_Rate"].mean()
    fig = px.line(data, x="Year", y="Interest_Rate", markers=True, title="Average Interest Rate by Year")
    fig.update_traces(line_width=3, marker_size=8)
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=25, r=25, t=70, b=35),
                      xaxis_title="Year", yaxis_title="Average interest rate")
    # Keep charts compact, with slimmer bars and consistent typography.
    fig.update_layout(
        height=320,
        font=dict(size=11),
        title_font=dict(size=14),
        margin=dict(l=35, r=25, t=65, b=55),
    )
    fig.update_traces(width=0.32, selector=dict(type="bar"))
    st.plotly_chart(fig, use_container_width=True)

    fig = px.histogram(df, x="Risk_Score", nbins=25, title="Risk Score Distribution")
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=25, r=25, t=70, b=35),
                      xaxis_title="Risk score", yaxis_title="Number of loans")
    # Keep charts compact, with slimmer bars and consistent typography.
    fig.update_layout(
        height=320,
        font=dict(size=11),
        title_font=dict(size=14),
        margin=dict(l=35, r=25, t=65, b=55),
    )
    fig.update_traces(width=0.32, selector=dict(type="bar"))
    st.plotly_chart(fig, use_container_width=True)

    fig = px.scatter(df, x="Credit_Score", y="Debt_to_Income_Ratio", color="Default_Status",
                     hover_data=["Loan_ID", "Loan_Type", "Loan_Amount"], title="Credit Score vs Debt-to-Income Ratio")
    fig.update_layout(template="plotly_white", height=480, margin=dict(l=25, r=25, t=70, b=35),
                      xaxis_title="Credit score", yaxis_title="Debt-to-income ratio")
    # Keep charts compact, with slimmer bars and consistent typography.
    fig.update_layout(
        height=320,
        font=dict(size=11),
        title_font=dict(size=14),
        margin=dict(l=35, r=25, t=65, b=55),
    )
    fig.update_traces(width=0.32, selector=dict(type="bar"))
    st.plotly_chart(fig, use_container_width=True)


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

    # Convert monthly income into annual income in LPA for display only.
    # The original Monthly_Income column remains unchanged for model calculations.
    display_df = display_df.copy()
    if "Monthly_Income" in display_df.columns:
        display_df["Annual Income (LPA)"] = (pd.to_numeric(display_df["Monthly_Income"], errors="coerce") * 12 / 100000).round(2)
        display_df.drop(columns=["Monthly_Income"], inplace=True)

    columns = [
        "Loan_ID", "Loan_Type", "Age_Group", "Employment_Type",
        "Annual Income (LPA)", "Loan_Amount", "Tenure_Months",
        "Interest_Rate", "EMI", "Credit_Score", "Previous_Default",
        "Existing_Loans", "Debt_to_Income_Ratio",
        "Loan_Approval_Date", "Outstanding_Amount",
        "Repayment_Status", "Default_Status"
    ]
    columns = [column for column in columns if column in display_df.columns]

    st.dataframe(
        display_df[columns],
        use_container_width=True,
        height=500
    )

    st.download_button(
        "⬇️ Download Transactions (Income in LPA)",
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

        age_order = ["21-30 years", "31-40 years", "41-50 years", "51-55 years"]
        available_ages = df["Age_Group"].dropna().astype(str).unique().tolist()
        age_options = sorted(available_ages, key=lambda value: (int(''.join(ch for ch in value.split('-')[0].split('–')[0] if ch.isdigit()) or 0), value))
        age_group = st.selectbox("Age Group (21–55 years)", age_options)

        employment = st.selectbox(
            "Employment Type",
            sorted(df["Employment_Type"].dropna().astype(str).unique())
        )

        previous_default = st.selectbox(
            "Previous Default",
            sorted(df["Previous_Default"].dropna().astype(str).unique())
        )

    with c2:

        income_lpa = st.number_input(
            "Annual Income (LPA)",
            min_value=0.0,
            value=3.60,
            step=0.25,
            help="Enter annual income in lakhs per annum. Example: 3.60 means ₹3,60,000 per year."
        )
        income = income_lpa * 100000 / 12  # Convert LPA to monthly income for the trained model.

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
            min_value=200,
            max_value=700,
            value=680,
            step=10
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

        # Keep charts compact, with slimmer bars and consistent typography.
    fig.update_layout(
        height=320,
        font=dict(size=11),
        title_font=dict(size=14),
        margin=dict(l=35, r=25, t=65, b=55),
    )
    fig.update_traces(width=0.32, selector=dict(type="bar"))
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
