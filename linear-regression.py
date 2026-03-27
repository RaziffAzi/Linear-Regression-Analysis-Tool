import streamlit as st
import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import f
import statsmodels.formula.api as smf
import statsmodels.api as sm

st.set_page_config(
    page_title="Linear Regression Analysis Tool",
    page_icon="📈",
    layout="centered",
    initial_sidebar_state="expanded",
)

st.title("Linear Regression Analysis Tool📈")
st.write('---')

st.sidebar.title('🔍 Navigation')
page = st.sidebar.radio('Go to', ['Data Input', 'Basic Statistics', 'Scatter Diagram', 'Regression Analysis'])
# Initialize session state for global DataFrame
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame()

@st.cache_data
def get_scatter_data(data, feature, target):
    return data[[feature, target]]

@st.cache_data
def get_residuals_df(residuals):
    return pd.DataFrame({'Residuals': residuals})

@st.cache_resource
def fit_linear_model(data):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    residuals = y - y_pred
    return model, y_pred, residuals

if page == 'Data Input':      
    
    option = st.selectbox(
        'What input data would you like to use?', 
        (
            'Enter Data Manually',      
            'Upload CSV File',        
            'Upload Excel File',      
            'Use Randomly Generated Data'
        )
    )

    if option == 'Enter Data Manually':
        st.header("Enter Data Manually")

        # Initialize empty DataFrame if not exists
        if "manual_data" not in st.session_state:
            st.session_state.manual_data = pd.DataFrame({
                "Feature1": np.array([1, 2]),
                "Feature2": np.array([3, 4]),
                "Target": np.array([5, 6])
            })

        # Add new column dynamically
        new_col_name = st.text_input("Enter new column name (leave empty if none):")
        if st.button("Add Column") and new_col_name:
            st.session_state.manual_data[new_col_name] = 0
            cols = st.session_state.manual_data.columns.tolist()
            cols[-1], cols[-2] = cols[-2], cols[-1]
            st.session_state.manual_data = st.session_state.manual_data[cols]
            st.success(f"Column '{new_col_name}' added!")

        # Editable table
        st.session_state.manual_data = st.data_editor(
            st.session_state.manual_data,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=False
        )

        # Load button
        if st.button("Load Data"):
            st.session_state.data = st.session_state.manual_data.copy()
            st.session_state.data = st.session_state.data.astype(float)
            st.success("Data loaded successfully!")
            st.dataframe(st.session_state.data)
    elif option == 'Upload CSV File':
        st.header("Upload CSV File")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded CSV:")
            st.dataframe(df)
            
            # Let user select X and Y columns
            all_columns = df.columns.tolist()
            y_column = st.selectbox("Select the Target (Y) column", all_columns)
            x_columns = st.multiselect("Select Predictor (X) columns", [c for c in all_columns if c != y_column])
            
            if st.button("Load CSV Data"):
                if x_columns and y_column:
                    st.session_state.data = df[x_columns + [y_column]]  # keep Xs and Y
                    st.success("CSV data loaded!")
                    st.dataframe(st.session_state.data)
                else:
                    st.warning("Please select at least one predictor (X) column and a target (Y) column.")
                    
    elif option == 'Upload Excel File':
        st.header("Upload Excel File")
        uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx", "xls"])
        if uploaded_file:
            df = pd.read_excel(uploaded_file)
            st.write("Preview of uploaded Excel:")
            st.dataframe(df)
            
            # Let user select X and Y columns
            all_columns = df.columns.tolist()
            y_column = st.selectbox("Select the Target (Y) column", all_columns)
            x_columns = st.multiselect("Select Predictor (X) columns", [c for c in all_columns if c != y_column])
            
            if st.button("Load Excel Data"):
                if x_columns and y_column:
                    st.session_state.data = df[x_columns + [y_column]]  # keep Xs and Y
                    st.success("Excel data loaded!")
                    st.dataframe(st.session_state.data)
                else:
                    st.warning("Please select at least one predictor (X) column and a target (Y) column.")

    elif option == 'Use Randomly Generated Data':
        st.header("Use Randomly Generated Data")
        num_rows = st.slider("Select number of rows", min_value=10, max_value=1000, value=100)

        # Initialize random data in session_state
        if "random_data" not in st.session_state:
            st.session_state.random_data = pd.DataFrame({
                "Feature1": np.random.rand(num_rows),
                "Feature2": np.random.rand(num_rows),
                "Target": np.random.rand(num_rows)
            })

        # Add new column dynamically
        new_col_name = st.text_input("Enter new column name (leave empty if none):")
        if st.button("Add Column (Random Data)") and new_col_name:
            st.session_state.random_data[new_col_name] = np.random.rand(num_rows)
            cols = st.session_state.random_data.columns.tolist()
            cols[-1], cols[-2] = cols[-2], cols[-1]
            st.session_state.random_data = st.session_state.random_data[cols]
            st.success(f"Column '{new_col_name}' added!")

        st.dataframe(st.session_state.random_data)

        # Load button
        if st.button("Load Random Data"):
            st.session_state.data = st.session_state.random_data.copy()
            st.success("Random data loaded!")

if page == 'Basic Statistics':
    st.header("Summary Statistics")
    if st.session_state.data.empty:
        st.info("No data available. Please provide data.")
    else:
        st.write(st.session_state.data.describe())

if page == 'Scatter Diagram':
    st.header("Scatter Diagram")

    if st.session_state.data.empty:
        st.info("No data available. Please provide data.")
    else:
        target_col = st.session_state.data.columns[-1]
        feature = st.selectbox("Select a feature column:", st.session_state.data.columns[:-1])

        scatter_df = get_scatter_data(st.session_state.data, feature, target_col)
        st.scatter_chart(scatter_df, x=feature, y=target_col)
        
if page == 'Regression Analysis':

    st.header("Regression Analysis")
    tab1, tab2, tab3 = st.tabs(["Equation & Fit", "ANOVA & P-values for Coefficients", "Residual Analysis"])

    # ===== TAB 1: Equation & Fit =====
    with tab1:
        if st.session_state.data.empty:
            st.info("No data available. Please provide data.")
        else:
            X = st.session_state.data.iloc[:, :-1].values
            y = st.session_state.data.iloc[:, -1].values

            model = LinearRegression()
            model.fit(X, y)

            st.subheader("📉 Regression Coefficients")
            coeff_df = pd.DataFrame({
                "Variable": st.session_state.data.columns[:-1],
                "Coefficient": model.coef_
            })
            st.dataframe(coeff_df.style.format({"Coefficient": "{:.4f}"}))

            st.subheader("⚙️ Intercept")
            st.write(f"{model.intercept_:.4f}")

            st.subheader("🔗 Correlation with Target")
            corr_df = pd.DataFrame({
                "Variable": st.session_state.data.columns[:-1],
                "Correlation": np.corrcoef(X.T, y)[-1, :-1]
            })
            st.dataframe(corr_df.style.format({"Correlation": "{:.4f}"}))

            st.subheader("🧮 Regression Equation")
            sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
            eq = f"Ŷ = {model.intercept_:.4f}"
            for i, coef in enumerate(model.coef_):
                sign = "+" if coef >= 0 else "-"
                eq += f" {sign} {abs(coef):.4f}·X{str(i+1).translate(sub)}"
            st.write(eq)

            st.subheader("📊 R-squared Value")
            r2 = model.score(X, y)
            st.write(f"R² = {r2:.4f}")

            if r2 >= 0.8:
                st.success("The model fits the data very well.")
            elif r2 >= 0.5:
                st.info("The model has a moderate fit.")
            else:
                st.warning("The model does not fit the data well.")

            st.subheader("📈 Predicted vs Actual Values")
            y_pred = model.predict(X)
            comparison_df = pd.DataFrame({"Actual": y, "Predicted": y_pred})
            st.dataframe(comparison_df.style.format("{:.4f}"))

    # ===== TAB 2: ANOVA & Coefficients =====
    with tab2:
        st.subheader("ANOVA & Coefficient Significance")
        if st.session_state.data.empty:
            st.info("No data available. Please provide data.")
        else:
            df = st.session_state.data.copy()

            # Safely quote all column names
            target_col = df.columns[-1]
            predictor_cols = df.columns[:-1]

            target = f"Q('{target_col}')"
            predictors = " + ".join([f"Q('{col}')" for col in predictor_cols])

            include_interaction = st.checkbox("Include Interaction Effects", value=False)
            if include_interaction and len(predictor_cols) > 1:
                formula = f"{target} ~ " + " * ".join([f"Q('{col}')" for col in predictor_cols])
            else:
                formula = f"{target} ~ {predictors}"

            st.caption("Model Formula Used:")
            st.code(formula)

            try:
                # Fit OLS model
                ols_model = smf.ols(formula, data=df).fit()

                # ANOVA Table
                st.subheader("📋 ANOVA Table")
                anova_table = sm.stats.anova_lm(ols_model, typ=2)
                st.dataframe(anova_table.style.format(precision=4))

                # Coefficients
                st.subheader("📊 Coefficients & P-values")
                coef_summary = pd.DataFrame({
                    'Variable': ols_model.params.index,
                    'Coefficient': ols_model.params.values,
                    'P-value': ols_model.pvalues.values,
                    'Std. Error': ols_model.bse.values
                })
                coef_summary = coef_summary[coef_summary['Variable'] != 'Intercept']
                st.dataframe(coef_summary.style.format({'Coefficient': '{:.4f}', 'P-value': '{:.4e}', 'Std. Error': '{:.4f}'}))

                # Significance interpretation
                st.subheader("🧠 Interpretation")
                sig = coef_summary[coef_summary['P-value'] < 0.05]
                nonsig = coef_summary[coef_summary['P-value'] >= 0.05]

                if not sig.empty:
                    st.success(f"✅ Significant variables: {', '.join(sig['Variable'])}")
                if not nonsig.empty:
                    st.warning(f"⚠️ Not significant: {', '.join(nonsig['Variable'])}")

            except Exception as e:
                st.error(f"An error occurred: {e}")

    # ===== TAB 3: Residual Analysis =====
    with tab3:
        st.subheader("Residuals")
        if st.session_state.data.empty:
            st.info("No data available. Please provide data.")
        else:
            # Use cached regression results to avoid refitting
            model, y_pred, residuals = fit_linear_model(st.session_state.data)

            residuals_df = get_residuals_df(residuals)

            # Display small summary first (faster to render than large table)
            st.write("**Summary of Residuals:**")
            st.write(residuals_df.describe().T.style.format("{:.4f}"))

            # Option to show full table (to avoid rendering large data unless needed)
            if st.checkbox("Show full residuals table"):
                st.dataframe(residuals_df.style.format("{:.4f}"), use_container_width=True)

            # Efficient charting (limit max rows to avoid slow rendering)
            max_display = min(len(residuals_df), 200)
            st.caption(f"Showing first {max_display} residuals for visualization:")
            st.bar_chart(residuals_df.head(max_display))

            st.info("Residuals represent the difference between actual and predicted values. "
                    "If most residuals are near zero, the model fits well.")
