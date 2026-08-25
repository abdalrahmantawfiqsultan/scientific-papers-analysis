import streamlit as st

def apply_custom_theme():
    """Injects a polished, modern dark academic theme with refined typography and spacing."""
    custom_css = """
    <style>
        /* ===== FONT: local-first, no external dependency ===== */

        /* ===== GLOBAL ===== */
        .stApp {
            background-color: #0C1222;
            color: #E2E8F0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* ===== SIDEBAR ===== */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #111827 0%, #0C1222 100%) !important;
            border-right: 1px solid rgba(99, 102, 241, 0.15);
        }

        /* ===== HEADERS ===== */
        h1 {
            font-weight: 700 !important;
            letter-spacing: -0.5px !important;
            color: #F1F5F9 !important;
        }

        h2, h3 {
            font-weight: 600 !important;
            color: #CBD5E1 !important;
            letter-spacing: -0.3px !important;
        }

        /* ===== TABS ===== */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background-color: transparent;
            padding: 0;
            border-bottom: 1px solid #1E293B;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 0;
            padding: 10px 16px;
            font-weight: 500;
            font-size: 14px;
            color: #94A3B8;
            background-color: transparent;
            border-bottom: 2px solid transparent;
            transition: color 0.2s ease;
        }

        .stTabs [data-baseweb="tab"]:hover {
            color: #E2E8F0;
        }

        .stTabs [aria-selected="true"] {
            color: #F1F5F9 !important;
            font-weight: 600;
            border-bottom: 2px solid #6366F1 !important;
            background-color: transparent !important;
        }

        /* ===== METRICS ===== */
        [data-testid="stMetricValue"] {
            font-size: 26px !important;
            font-weight: 700 !important;
            color: #F1F5F9 !important;
        }

        [data-testid="stMetricLabel"] {
            font-size: 12px !important;
            font-weight: 500 !important;
            color: #94A3B8 !important;
        }

        /* ===== BUTTONS ===== */
        /* Primary */
        .stButton > button[kind="primary"] {
            background-color: #6366F1;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
            font-size: 14px;
            transition: background-color 0.2s ease;
        }

        .stButton > button[kind="primary"]:hover {
            background-color: #818CF8;
            color: white;
        }
        
        .stButton > button[kind="primary"]:disabled {
            background-color: #334155;
            color: #64748B;
        }

        /* Secondary */
        .stButton > button[kind="secondary"] {
            background-color: #1E293B;
            border: 1px solid #334155;
            color: #CBD5E1;
            border-radius: 6px;
            padding: 8px 16px;
            font-weight: 500;
            font-size: 14px;
            transition: all 0.2s ease;
        }

        .stButton > button[kind="secondary"]:hover {
            background-color: #334155;
            border-color: #475569;
            color: #F1F5F9;
        }
        
        /* Destructive */
        .stButton.destructive > button {
            background-color: transparent;
            border: 1px solid #7F1D1D;
            color: #EF4444;
        }
        
        .stButton.destructive > button:hover {
            background-color: #7F1D1D;
            color: #FCA5A5;
        }

        /* ===== FILE UPLOADER ===== */
        [data-testid="stFileUploader"] {
            border: 1px dashed #475569;
            border-radius: 6px;
            padding: 24px;
            background-color: #0F172A;
            transition: border-color 0.2s ease;
        }

        [data-testid="stFileUploader"]:hover {
            border-color: #6366F1;
        }

        /* ===== TEXT INPUT ===== */
        .stTextInput > div > div > input {
            background-color: #0F172A !important;
            border: 1px solid #334155 !important;
            border-radius: 6px !important;
            color: #E2E8F0 !important;
            padding: 10px 12px !important;
            font-size: 14px !important;
        }

        .stTextInput > div > div > input:focus {
            border-color: #6366F1 !important;
            box-shadow: none !important;
            outline: 1px solid #6366F1;
        }

        /* ===== SELECT BOX ===== */
        .stSelectbox > div > div {
            background-color: #0F172A !important;
            border: 1px solid #334155 !important;
            border-radius: 6px !important;
        }

        /* ===== EXPANDER / STATUS ===== */
        .streamlit-expanderHeader {
            font-weight: 500 !important;
            color: #CBD5E1 !important;
            background-color: transparent !important;
            border-radius: 0 !important;
            border-bottom: 1px solid #1E293B !important;
        }

        /* ===== DIVIDER ===== */
        hr {
            border-color: #1E293B !important;
            opacity: 0.5;
        }

        /* ===== RADIO BUTTONS ===== */
        .stRadio > div {
            gap: 4px;
        }

        .stRadio > div > label {
            padding: 6px 12px;
            border-radius: 6px;
            transition: background-color 0.2s ease;
        }

        .stRadio > div > label:hover {
            background-color: rgba(99, 102, 241, 0.08);
        }

        /* ===== CUSTOM CARDS ===== */
        .hypothesis-card {
            background-color: #111827;
            border: 1px solid #1E293B;
            border-left: 3px solid #6366F1;
            border-radius: 4px;
            padding: 16px;
            margin-bottom: 20px;
        }

        /* ===== METRIC BADGES ===== */
        .metric-badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            margin-right: 8px;
        }

        .badge-method { background-color: rgba(16, 185, 129, 0.15); color: #34D399; border: 1px solid rgba(16, 185, 129, 0.3); }
        .badge-concept { background-color: rgba(245, 158, 11, 0.15); color: #FBBF24; border: 1px solid rgba(245, 158, 11, 0.3); }
        .badge-score { background-color: rgba(99, 102, 241, 0.15); color: #A5B4FC; border: 1px solid rgba(99, 102, 241, 0.3); }
        .badge-paper { background-color: rgba(56, 189, 248, 0.15); color: #7DD3FC; border: 1px solid rgba(56, 189, 248, 0.3); }
        .badge-author { background-color: rgba(236, 72, 153, 0.15); color: #F9A8D4; border: 1px solid rgba(236, 72, 153, 0.3); }

        /* ===== STEPPER ===== */
        .step-container {
            display: flex;
            justify-content: space-between;
            margin-bottom: 25px;
            background: linear-gradient(135deg, #111827, #1E293B);
            padding: 14px 20px;
            border-radius: 10px;
            border: 1px solid #1E293B;
        }

        .step-item {
            display: flex;
            align-items: center;
            font-size: 13px;
            font-weight: 500;
            color: #64748B;
        }

        .step-active { color: #818CF8; font-weight: 700; }
        .step-completed { color: #34D399; }

        /* ===== GRAPH STATS FOOTER ===== */
        .graph-stats {
            display: flex;
            gap: 16px;
            padding: 10px 16px;
            background-color: rgba(17, 24, 39, 0.8);
            border-radius: 8px;
            border: 1px solid #1E293B;
            margin-top: 8px;
        }

        .graph-stats span {
            font-size: 12px;
            color: #64748B;
            font-weight: 500;
        }

        .graph-stats span b {
            color: #A5B4FC;
        }

        /* ===== TOAST / ALERTS ===== */
        .stAlert {
            border-radius: 10px !important;
        }

        /* ===== SCROLLBAR ===== */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #0C1222;
        }
        ::-webkit-scrollbar-thumb {
            background: #334155;
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #475569;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)
