import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)


@st.cache_data
def generate_sample_data():
    """Generate synthetic telco customer data for training"""
    np.random.seed(42)
    n_samples = 1000

    contract_lengths = np.random.choice([1, 12, 24], n_samples, p=[0.3, 0.4, 0.3])
    monthly_charges = np.random.normal(65, 20, n_samples)
    monthly_charges = np.clip(monthly_charges, 20, 120)  # Realistic range
    payment_history = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])  # 0=good, 1=avg, 2=poor
    tenure = np.random.exponential(24, n_samples)  # months with company
    tenure = np.clip(tenure, 1, 72)

    # Create churn labels with realistic logic
    churn_prob = (
            0.1 +  # Base churn rate
            0.4 * (contract_lengths == 1) +  # Month-to-month more likely to churn
            0.2 * (monthly_charges > 80) / 100 +  # Higher charges increase churn
            0.3 * (payment_history == 2) +  # Poor payment history increases churn
            0.2 * (payment_history == 1) +  # Average payment history moderate increase
            -0.2 * (tenure > 24) / 72  # Longer tenure reduces churn
    )

    churn_prob = np.clip(churn_prob, 0, 1)
    churn = np.random.binomial(1, churn_prob, n_samples)

    df = pd.DataFrame({
        'contract_length': contract_lengths,
        'monthly_charges': monthly_charges,
        'payment_history': payment_history,
        'tenure': tenure,
        'churn': churn
    })

    return df


@st.cache_resource
def train_model():
    """Train the Random Forest model"""
    # Generate training data
    df = generate_sample_data()

    # Prepare features and target
    X = df[['contract_length', 'monthly_charges', 'payment_history', 'tenure']]
    y = df['churn']

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X, y)

    return model


def predict_churn(model, contract_length, monthly_charges, payment_history, tenure=24):
    """Make churn prediction"""
    # Encode payment history
    payment_map = {'Good': 0, 'Average': 1, 'Poor': 2}
    payment_encoded = payment_map[payment_history]

    # Create input array
    input_data = np.array([[contract_length, monthly_charges, payment_encoded, tenure]])

    # Get prediction and probability
    churn_prob = model.predict_proba(input_data)[0][1]  # Probability of churn (class 1)
    churn_prediction = model.predict(input_data)[0]

    return churn_prob, churn_prediction


def main():
    st.title("📊 Customer Churn Prediction")
    st.markdown("Predict whether a customer is likely to leave based on their profile")

    # Load/train model
    with st.spinner("Loading ML model..."):
        model = train_model()

    st.markdown("---")

    # Input form
    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        contract_length = st.selectbox(
            "Contract Length",
            options=[1, 12, 24],
            format_func=lambda x: f"{x} month{'s' if x > 1 else ''}",
            help="Customer's current contract duration"
        )

        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=20.0,
            max_value=120.0,
            value=65.0,
            step=5.0,
            help="Customer's monthly bill amount"
        )

    with col2:
        payment_history = st.selectbox(
            "Payment History",
            options=["Good", "Average", "Poor"],
            help="Customer's payment track record"
        )

        tenure = st.slider(
            "Tenure (months)",
            min_value=1,
            max_value=72,
            value=24,
            help="How long customer has been with the company"
        )

    # Predict button
    if st.button("Predict Churn", type="primary"):
        with st.spinner("Analyzing customer data..."):
            churn_prob, churn_prediction = predict_churn(
                model, contract_length, monthly_charges, payment_history, tenure
            )

        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")

        # Create result columns
        col1, col2 = st.columns(2)

        with col1:
            # Churn probability
            risk_percentage = int(churn_prob * 100)
            st.metric(
                label="Churn Risk",
                value=f"{risk_percentage}%"
            )

            # Progress bar for visual representation
            st.progress(churn_prob)

        with col2:
            # Prediction result
            if churn_prediction == 1:
                st.error("🚨 **Likely to Leave**")
                recommendation = "High risk customer. Consider retention strategies."
            else:
                st.success("✅ **Likely to Stay**")
                recommendation = "Low risk customer. Maintain current service level."

            st.write(f"*{recommendation}*")

        # Risk level interpretation
        if risk_percentage >= 70:
            risk_level = "🔴 **High Risk**"
            risk_color = "red"
        elif risk_percentage >= 40:
            risk_level = "🟡 **Medium Risk**"
            risk_color = "orange"
        else:
            risk_level = "🟢 **Low Risk**"
            risk_color = "green"

        st.markdown(f"**Risk Level:** {risk_level}")

        # Feature importance insights
        with st.expander("📈 See Model Insights"):
            feature_importance = model.feature_importances_
            features = ['Contract Length', 'Monthly Charges', 'Payment History', 'Tenure']

            importance_df = pd.DataFrame({
                'Feature': features,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)

            st.bar_chart(importance_df.set_index('Feature'))
            st.caption("Feature importance shows which factors most influence churn predictions")

    # Sidebar with information
    st.sidebar.markdown("## About This Tool")
    st.sidebar.info(
        """
        This MVP uses a Random Forest classifier trained on synthetic customer data 
        to predict churn probability.

        **Key Features:**
        - Contract length
        - Monthly charges
        - Payment history
        - Customer tenure

        **Model:** Random Forest with 100 trees
        """
    )

    st.sidebar.markdown("## Sample Scenarios")
    st.sidebar.markdown(
        """
        **High Risk:**
        - Month-to-month contract
        - High monthly charges
        - Poor payment history

        **Low Risk:**
        - 24-month contract
        - Moderate charges
        - Good payment history
        - Long tenure
        """
    )


if __name__ == "__main__":
    main()