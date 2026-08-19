import streamlit as st

def apply_custom_theme():
    """Injects custom CSS to achieve a modern academic dark telemetry design."""
    custom_css = """
    <style>
        /* Main Application Background */
        .stApp {
            background-color: #0F172A;
            color: #F8FAFC;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #1E293B !important;
            border-right: 1px solid #334155;
        }

        /* Custom Hypothesis & Evidence Cards */
        .hypothesis-card {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
            border: 1px solid #38BDF8;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 14px rgba(56, 189, 248, 0.1);
        }

        .metric-badge {
            display: inline-block;
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 600;
            margin-right: 8px;
        }

        .badge-method { background-color: rgba(16, 185, 129, 0.2); color: #34D399; border: 1px solid #10B981; }
        .badge-concept { background-color: rgba(245, 158, 11, 0.2); color: #FBBF24; border: 1px solid #F59E0B; }
        .badge-score { background-color: rgba(59, 130, 246, 0.2); color: #60A5FA; border: 1px solid #3B82F6; }

        /* Stepper Component */
        .step-container {
            display: flex;
            justify-content: space-between;
            margin-bottom: 25px;
            background-color: #1E293B;
            padding: 12px;
            border-radius: 8px;
            border: 1px solid #334155;
        }

        .step-item {
            display: flex;
            align-items: center;
            font-size: 13px;
            font-weight: 500;
            color: #94A3B8;
        }

        .step-active { color: #38BDF8; font-weight: 700; }
        .step-completed { color: #34D399; }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
