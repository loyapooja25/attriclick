# Final Streamlit App with All Fixes and Clean Logic
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Load models
attr_model = joblib.load("best_attrition_mmmodelaa-aa_cccatboostt.pkl")
perf_model = joblib.load("catboost_best_modell-b.pkl")

# Load and process training data for reasoning
@st.cache_data
def load_reasoning_data():
    df = pd.read_csv("WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df = df.drop(columns=["EmployeeCount", "EmployeeNumber", "StandardHours", "Over18"])
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    leavers = df[df['Attrition'] == 1]
    stayers = df[df['Attrition'] == 0]

    reasoning = {}
    for col in df.columns:
        if col != 'Attrition':
            leaver_avg = leavers[col].mean()
            stayer_avg = stayers[col].mean()
            diff = leaver_avg - stayer_avg
            if abs(diff) > 0.2:
                reasoning[col] = {
                    "suggestion": f"Focus on improving or balancing '{col}' to better align with retained employee characteristics."
                }
    return reasoning

reasoning_map = load_reasoning_data()

# Reason and suggestion generator
def explain_reason(row):
    reasons, suggestions = [], []
    for key in row.keys():
        if key in reasoning_map:
            reasons.append(f"{key} may be contributing to attrition based on historical data.")
            suggestions.append(reasoning_map[key]['suggestion'])
    return reasons, suggestions

# Encoded mappings
business_travel_map = {"Travel_Rarely": 1, "Travel_Frequently": 2, "Non-Travel": 0}
department_map = {"Sales": 2, "Research & Development": 1, "Human Resources": 0}
education_field_map = {
    "Life Sciences": 4, "Medical": 2, "Marketing": 3,
    "Technical Degree": 5, "Other": 1, "Human Resources": 0
}
job_role_map = {
    "Sales Executive": 6, "Research Scientist": 7, "Laboratory Technician": 2,
    "Manufacturing Director": 1, "Healthcare Representative": 3, "Manager": 4,
    "Sales Representative": 8, "Research Director": 5, "Human Resources": 0
}
marital_status_map = {"Married": 0, "Single": 1, "Divorced": 2}
gender_map = {"Male": 1, "Female": 0}
overtime_map = {"Yes": 1, "No": 0}

st.title("Attrition & Performance Risk Analyzer")
option = st.radio("Choose Mode", ["Single Prediction", "Upload CSV for Bulk Prediction"])

# Single Prediction Mode
if option == "Single Prediction":
    st.header("Enter Employee Details")
    age = st.slider("Age", 18, 60, 34)
    business_travel = st.selectbox("Business Travel", list(business_travel_map.keys()))
    daily_rate = st.slider("Daily Rate", 100, 1500, 800)
    department = st.selectbox("Department", list(department_map.keys()))
    distance = st.slider("Distance From Home", 1, 30, 4)
    education = st.slider("Education Level", 1, 5, 3)
    education_field = st.selectbox("Education Field", list(education_field_map.keys()))
    env_satisfaction = st.slider("Environment Satisfaction", 1, 4, 2)
    gender = st.selectbox("Gender", list(gender_map.keys()))
    hourly_rate = st.slider("Hourly Rate", 40, 100, 70)
    job_involvement = st.slider("Job Involvement", 1, 4, 3)
    job_level = st.slider("Job Level", 1, 5, 2)
    job_role = st.selectbox("Job Role", list(job_role_map.keys()))
    job_satisfaction = st.slider("Job Satisfaction", 1, 4, 3)
    marital_status = st.selectbox("Marital Status", list(marital_status_map.keys()))
    monthly_income = st.slider("Monthly Income", 1000, 20000, 6500)
    monthly_rate = st.slider("Monthly Rate", 1000, 30000, 14000)
    num_companies = st.slider("Num Companies Worked", 0, 10, 2)
    overtime = st.selectbox("OverTime", list(overtime_map.keys()))
    percent_hike = st.slider("Percent Salary Hike", 10, 30, 15)
    relationship_satisfaction = st.slider("Relationship Satisfaction", 1, 4, 2)
    stock_option = st.slider("Stock Option Level", 0, 3, 1)
    total_years = st.slider("Total Working Years", 0, 40, 10)
    training = st.slider("Training Times Last Year", 0, 6, 3)
    wlb = st.slider("Work Life Balance", 1, 4, 2)
    years_company = st.slider("Years at Company", 0, 40, 5)
    years_role = st.slider("Years in Current Role", 0, 20, 2)
    years_promotion = st.slider("Years Since Last Promotion", 0, 15, 1)
    years_manager = st.slider("Years With Current Manager", 0, 20, 3)

    attr_input = pd.DataFrame([{
        "Age": age,
        "BusinessTravel": business_travel_map[business_travel],
        "DailyRate": daily_rate,
        "Department": department_map[department],
        "DistanceFromHome": distance,
        "Education": education,
        "EducationField": education_field_map[education_field],
        "EnvironmentSatisfaction": env_satisfaction,
        "Gender": gender_map[gender],
        "HourlyRate": hourly_rate,
        "JobInvolvement": job_involvement,
        "JobLevel": job_level,
        "JobRole": job_role_map[job_role],
        "JobSatisfaction": job_satisfaction,
        "MaritalStatus": marital_status_map[marital_status],
        "MonthlyIncome": monthly_income,
        "MonthlyRate": monthly_rate,
        "NumCompaniesWorked": num_companies,
        "OverTime": overtime_map[overtime],
        "PercentSalaryHike": percent_hike,
        "RelationshipSatisfaction": relationship_satisfaction,
        "StockOptionLevel": stock_option,
        "TotalWorkingYears": total_years,
        "TrainingTimesLastYear": training,
        "WorkLifeBalance": wlb,
        "YearsAtCompany": years_company,
        "YearsInCurrentRole": years_role,
        "YearsSinceLastPromotion": years_promotion,
        "YearsWithCurrManager": years_manager
    }])

    perf_input = pd.DataFrame([{
        "EmpEnvironmentSatisfaction": env_satisfaction,
        "EmpLastSalaryHikePercent": percent_hike,
        "EmpWorkLifeBalance": wlb,
        "ExperienceYearsAtThisCompany": years_company,
        "YearsWithCurrManager": years_manager,
        "ExperienceYearsInCurrentRole": years_role,
        "YearsSinceLastPromotion": years_promotion
    }])

    if st.button("Predict"):
        attr = attr_model.predict(attr_input)[0]
        perf = perf_model.predict(perf_input)[0]
        st.subheader("Prediction Results")
        st.write("Attrition Risk:", "Yes" if attr == 1 else "No")
        st.write("Performance Rating:", perf)

        if attr == 1 and perf >= 3:
            st.warning("High-performing employee at risk")
        elif attr == 1:
            st.warning("Likely to leave")
        else:
            st.success("No immediate attrition risk")

        st.subheader("Data-Inferred Reasons and Retention Suggestions")
        reasons, suggestions = explain_reason(attr_input.iloc[0])
        if reasons:
            st.write("Reasons:")
            for r in reasons:
                st.write("-", r)
            st.write("Recommendations:")
            for s in suggestions:
                st.write("-", s)
        else:
            st.write("No clear risks identified from historical patterns.")

# Bulk CSV Upload
if option == "Upload CSV for Bulk Prediction":
    file = st.file_uploader("Upload file with employee data", type=["csv"])
    if file:
        df = pd.read_csv(file)
        attr_preds = attr_model.predict(df)
        perf_preds = perf_model.predict(df)

        df["Attrition_Prediction"] = attr_preds
        df["Performance_Prediction"] = perf_preds

        reasons_list, suggestions_list, flags = [], [], []
        for _, row in df.iterrows():
            reasons, suggestions = explain_reason(row)
            reasons_list.append(" | ".join(reasons))
            suggestions_list.append(" | ".join(suggestions))
            if row['Attrition_Prediction'] == 1 and row['Performance_Prediction'] >= 3:
                flags.append("High-performing employee at risk")
            elif row['Attrition_Prediction'] == 1:
                flags.append("Likely to leave")
            else:
                flags.append("No immediate attrition risk")

        df["Reasons"] = reasons_list
        df["Suggestions"] = suggestions_list
        df["Risk_Flag"] = flags

        st.subheader("Batch Prediction Results")
        st.dataframe(df)
        st.download_button("Download Results", df.to_csv(index=False), file_name="prediction_results.csv")
