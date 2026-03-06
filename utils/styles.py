import streamlit as st

def inject_global_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');

    :root {
        /* Dark theme colors */
        --bg-body: #0f172a;
        --bg-surface: #1e293b;
        --bg-surface-light: #334155;
        --bg-surface-hover: #2d3a4f;
        --bg-gradient-start: #3b82f6;
        --bg-gradient-end: #8b5cf6;
        --bg-gradient-card: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        
        /* Text colors - all bright and visible */
        --text-primary: #f1f5f9;
        --text-secondary: #cbd5e1;
        --text-tertiary: #94a3b8;
        --text-on-gradient: #ffffff;
        --text-light: #f8fafc;
        --text-dark: #0f172a;
        
        /* Accent colors - all blue variants */
        --accent: #3b82f6;
        --accent-light: #2563eb;
        --accent-primary: #3b82f6;
        --accent-secondary: #60a5fa;
        --accent-success: #3b82f6;
        --accent-warning: #3b82f6;
        --accent-error: #ef4444;
        --accent-info: #3b82f6;
        
        /* UI Elements */
        --border-light: #334155;
        --border-medium: #475569;
        --shadow-sm: 0 2px 8px rgba(0,0,0,0.5);
        --shadow-md: 0 4px 20px rgba(0,0,0,0.6);
        --shadow-lg: 0 10px 30px rgba(0,0,0,0.7);
        --shadow-xl: 0 20px 40px rgba(0,0,0,0.8);
        --glow: 0 0 15px rgba(59,130,246,0.3);
    }

    /* Base styles */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stApp"], .main {
        background: var(--bg-body) !important;
        color: var(--text-primary) !important;
        font-family: 'Inter', sans-serif !important;
    }

    [data-testid="stMainBlockContainer"] {
        background: var(--bg-body) !important;
        max-width: 1400px !important;
        padding: 2rem 2rem !important;
        margin: 0 auto !important;
    }

    /* Hide default Streamlit elements */
    #MainMenu, footer, header,
    [data-testid="stToolbar"],
    [data-testid="stDecoration"],
    [data-testid="stStatusWidget"] { 
        display: none !important; 
    }

    /* Dashboard Hero Section - Dark with gradient */
    .dashboard-hero {
        text-align: center;
        padding: 3rem 2rem;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 32px;
        border: 1px solid var(--border-light);
        box-shadow: var(--shadow-lg);
    }

    .hero-title {
        font-size: 4rem;
        font-weight: 800;
        letter-spacing: -2px;
        background: linear-gradient(135deg, #3b82f6 0%, #93c5fd 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        font-family: 'Space Grotesk', sans-serif;
        animation: fadeInUp 0.8s ease;
        text-shadow: 0 2px 10px rgba(59, 130, 246, 0.3);
    }

    .hero-subtitle {
        font-size: 1.25rem;
        color: var(--text-secondary);
        max-width: 600px;
        margin: 0 auto 2rem;
        line-height: 1.6;
        animation: fadeInUp 1s ease;
    }

    /* Feature Cards - Dark theme */
    .features-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1.5rem;
        margin: 2rem 0;
    }

    .feature-card {
        background: var(--bg-surface);
        border-radius: 24px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid var(--border-light);
        box-shadow: var(--shadow-sm);
        transition: none;
        animation: fadeInUp 1.2s ease;
    }

    .feature-card:hover {
        /* Hover effect removed - no changes */
    }

    .feature-emoji {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        display: inline-block;
        background: #1e3a8a;
        padding: 1rem;
        border-radius: 20px;
    }

    .feature-title {
        font-size: 1.1rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: var(--text-primary);
    }

    .feature-desc {
        font-size: 0.9rem;
        color: var(--text-secondary);
        line-height: 1.5;
    }

    /* CTA Section - Bright blue gradient */
    .cta-section {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        border-radius: 32px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 3rem 0;
        box-shadow: var(--shadow-lg);
        animation: fadeInUp 1.4s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .cta-title {
        color: white;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        font-family: 'Space Grotesk', sans-serif;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    .cta-subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }

    /* Welcome Section for Auth Users - Blue gradient */
    .welcome-section {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        border-radius: 32px;
        padding: 3rem 2rem;
        text-align: center;
        margin: 3rem 0;
        box-shadow: var(--shadow-lg);
        animation: fadeInUp 1.4s ease;
        border: 1px solid rgba(255,255,255,0.1);
    }

    .welcome-title {
        color: white;
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
        font-family: 'Space Grotesk', sans-serif;
        text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }

    .welcome-text {
        color: rgba(255,255,255,0.95);
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }

    /* Custom Buttons - All in blue */
    .dashboard-btn {
        background: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 999px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3) !important;
        transition: none !important;
        width: auto !important;
        min-width: 200px !important;
        margin: 0 auto !important;
    }

    .dashboard-btn:hover {
        /* Hover effect removed */
    }

    .dashboard-btn-primary {
        background: #2563eb !important;
        color: white !important;
        border: none !important;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3) !important;
    }

    .dashboard-btn-primary:hover {
        /* Hover effect removed */
    }

    /* Stats Section - Dark */
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1.5rem;
        margin: 2rem 0;
    }

    .stat-card {
        background: var(--bg-surface);
        border-radius: 20px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid var(--border-light);
        box-shadow: var(--shadow-sm);
    }

    .stat-number {
        font-size: 2rem;
        font-weight: 800;
        color: #3b82f6;
        margin-bottom: 0.25rem;
    }

    .stat-label {
        font-size: 0.9rem;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* ===== COMPACT AUTHENTICATION FORM STYLES - DARK THEME ===== */
    .auth-wrapper {
        max-width: 360px;
        margin: 1rem auto;
        background: var(--bg-surface);
        border-radius: 24px;
        box-shadow: var(--shadow-xl);
        overflow: hidden;
        animation: slideIn 0.6s ease;
        border: 1px solid var(--border-light);
    }

    .auth-header {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        padding: 1.2rem 1.5rem 0.8rem;
        text-align: center;
        color: white;
    }

    .auth-header h2 {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        font-family: 'Space Grotesk', sans-serif;
        color: white;
    }

    .auth-header p {
        font-size: 0.8rem;
        opacity: 0.9;
        margin-bottom: 0;
        color: rgba(255,255,255,0.9);
    }

    .auth-body {
        padding: 1.2rem 1.5rem;
        background: var(--bg-surface);
    }

    /* Form fields - Dark theme */
    .auth-field {
        margin-bottom: 1rem;
    }

    .auth-field label {
        display: block;
        font-size: 0.75rem;
        font-weight: 600;
        color: var(--text-secondary);
        margin-bottom: 0.25rem;
        letter-spacing: 0.3px;
    }

    /* Streamlit input overrides - Dark theme */
    [data-testid="stTextInput"],
    [data-testid="stPasswordInput"],
    [data-testid="stTextArea"] {
        margin-bottom: 0 !important;
    }

    [data-testid="stTextInput"] > div,
    [data-testid="stPasswordInput"] > div,
    [data-testid="stTextArea"] > div {
        margin-bottom: 0 !important;
    }

    [data-testid="stTextInput"] input,
    [data-testid="stPasswordInput"] input,
    [data-testid="stTextArea"] textarea {
        background: var(--bg-body) !important;
        border: 1.5px solid var(--border-medium) !important;
        border-radius: 12px !important;
        padding: 0.6rem 1rem !important;
        font-size: 0.85rem !important;
        color: var(--text-primary) !important;
        transition: none !important;
        box-shadow: none !important;
    }

    [data-testid="stTextInput"] input:hover,
    [data-testid="stPasswordInput"] input:hover,
    [data-testid="stTextArea"] textarea:hover {
        /* Hover effect removed */
    }

    [data-testid="stTextInput"] input:focus,
    [data-testid="stPasswordInput"] input:focus,
    [data-testid="stTextArea"] textarea:focus {
        border-color: #3b82f6 !important;
        background: var(--bg-surface-hover) !important;
        box-shadow: none !important;
    }

    [data-testid="stTextInput"] input::placeholder,
    [data-testid="stPasswordInput"] input::placeholder,
    [data-testid="stTextArea"] textarea::placeholder {
        color: var(--text-tertiary) !important;
        font-size: 0.8rem;
    }

    [data-testid="stTextInput"] label,
    [data-testid="stPasswordInput"] label,
    [data-testid="stTextArea"] label {
        display: none !important;
    }

    /* Google button - Blue theme */
    .google-btn {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 8px !important;
        width: 100% !important;
        padding: 0.6rem !important;
        background: #2563eb !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        text-decoration: none !important;
        transition: none !important;
        margin-bottom: 0.8rem !important;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3) !important;
    }

    .google-btn:hover {
        /* Hover effect removed */
    }

    .google-btn img {
        width: 16px !important;
        height: 16px !important;
        filter: brightness(1.2);
    }

    /* Divider - Dark theme */
    .auth-divider {
        display: flex;
        align-items: center;
        gap: 0.8rem;
        margin: 1rem 0;
    }

    .auth-divider-line {
        flex: 1;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-medium), transparent);
    }

    .auth-divider-text {
        color: var(--text-tertiary);
        font-size: 0.7rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Submit button - Solid blue */
    .auth-submit-btn {
        width: 100% !important;
        background: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem !important;
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        cursor: pointer !important;
        transition: none !important;
        margin-top: 0.8rem !important;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3) !important;
    }

    .auth-submit-btn:hover {
        /* Hover effect removed */
    }

    .auth-submit-btn:active {
        /* No change */
    }

    /* Auth footer - Dark theme */
    .auth-footer {
        text-align: center;
        margin-top: 1.2rem;
        padding-top: 1rem;
        border-top: 1px solid var(--border-light);
    }

    .auth-footer p {
        color: var(--text-secondary);
        font-size: 0.8rem;
        margin-bottom: 0;
    }

    .auth-footer a {
        color: #3b82f6;
        font-weight: 600;
        text-decoration: none;
        font-size: 0.8rem;
        transition: none;
    }

    .auth-footer a:hover {
        /* Hover effect removed */
    }

    /* Password strength indicator - Dark theme */
    .password-strength {
        margin-top: 0.3rem;
    }

    .strength-bars {
        display: flex;
        gap: 3px;
        margin-bottom: 0.2rem;
    }

    .strength-bar {
        flex: 1;
        height: 3px;
        background: var(--border-medium);
        border-radius: 999px;
        transition: none;
    }

    .strength-bar.active {
        background: #3b82f6;
    }

    .strength-text {
        font-size: 0.7rem;
        color: var(--text-secondary);
        text-align: right;
    }

    /* Alert styling - Dark theme */
    [data-testid="stAlert"] {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 12px !important;
        padding: 0.7rem !important;
        color: #fecaca !important;
        font-size: 0.8rem !important;
        margin-bottom: 0.8rem !important;
    }

    [data-testid="stAlert"] [data-testid="stAlertContent"] {
        color: #fecaca !important;
    }

    /* Success message - Blue theme */
    .success-message {
        background: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 12px !important;
        padding: 0.7rem !important;
        color: #bfdbfe !important;
        font-size: 0.8rem !important;
        margin-bottom: 0.8rem !important;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Tabs styling - Dark theme with blue accent */
    .auth-tabs {
        margin-bottom: 0.5rem;
    }

    .auth-tabs .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
        border-bottom: 1px solid var(--border-light);
        margin-bottom: 1rem;
        padding: 0;
    }

    .auth-tabs .stTabs [data-baseweb="tab"] {
        height: 2.2rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text-secondary);
        background: transparent;
        border: none;
        padding: 0 0.5rem;
        transition: none;
    }

    .auth-tabs .stTabs [data-baseweb="tab"]:hover {
        /* Hover effect removed */
    }

    .auth-tabs .stTabs [aria-selected="true"] {
        color: #3b82f6 !important;
        border-bottom: 2px solid #3b82f6 !important;
    }

    /* Select boxes - Dark theme with blue accent */
    [data-testid="stSelectbox"] {
        color: var(--text-primary) !important;
    }

    [data-testid="stSelectbox"] > div {
        background: var(--bg-body) !important;
        border: 1px solid var(--border-medium) !important;
        border-radius: 12px !important;
    }

    [data-testid="stSelectbox"]:hover > div {
        /* Hover effect removed */
    }

    /* Radio buttons - Dark theme */
    [data-testid="stRadio"] {
        color: var(--text-primary) !important;
    }

    [data-testid="stRadio"] label {
        color: var(--text-secondary) !important;
    }

    [data-testid="stRadio"] input:checked + div {
        color: #3b82f6 !important;
    }

    /* Checkbox - Dark theme */
    [data-testid="stCheckbox"] {
        color: var(--text-primary) !important;
    }

    [data-testid="stCheckbox"] label {
        color: var(--text-secondary) !important;
    }

    [data-testid="stCheckbox"] input:checked + div {
        color: #3b82f6 !important;
    }

    /* Sliders - Dark theme with blue accent */
    [data-testid="stSlider"] {
        color: var(--text-primary) !important;
    }

    [data-testid="stSlider"] div[data-baseweb="slider"] {
        background: var(--border-medium) !important;
    }

    [data-testid="stSlider"] div[role="slider"] {
        background: #3b82f6 !important;
    }

    /* Download button - Blue */
    [data-testid="stDownloadButton"] button {
        background: #2563eb !important;
        color: white !important;
        border: none !important;
        border-radius: 999px !important;
        padding: 0.75rem !important;
        font-weight: 600 !important;
        transition: none !important;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.3) !important;
    }

    [data-testid="stDownloadButton"] button:hover {
        /* Hover effect removed */
    }

    /* Progress bar - Blue */
    [data-testid="stProgress"] > div {
        background: var(--border-medium) !important;
    }

    [data-testid="stProgress"] > div > div {
        background: #3b82f6 !important;
    }

    /* Spinner - Blue */
    .stSpinner {
        border-color: #3b82f6 !important;
    }

    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }

    /* Custom scrollbar for dark theme */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-body);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--border-medium);
        border-radius: 4px;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .hero-title {
            font-size: 2.5rem;
        }
        
        .features-grid {
            grid-template-columns: repeat(2, 1fr);
        }
        
        .stats-grid {
            grid-template-columns: 1fr;
        }

        .auth-wrapper {
            margin: 0.5rem;
            max-width: 100%;
        }
    }

    /* Additional dark theme enhancements */
    hr {
        border-color: var(--border-light) !important;
    }

    a {
        color: #3b82f6 !important;
    }

    a:hover {
        /* Hover effect removed */
    }

    code {
        background: var(--bg-surface) !important;
        color: #3b82f6 !important;
        border: 1px solid var(--border-light) !important;
    }

    pre {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-light) !important;
    }

    /* Tooltips */
    [data-testid="stTooltip"] {
        background: var(--bg-surface-light) !important;
        color: var(--text-primary) !important;
        border: 1px solid var(--border-light) !important;
    }

    /* Expander */
    [data-testid="stExpander"] {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-light) !important;
        border-radius: 12px !important;
    }

    [data-testid="stExpander"] summary {
        color: var(--text-primary) !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: var(--bg-surface) !important;
        border: 1px dashed var(--border-medium) !important;
        border-radius: 12px !important;
    }

    [data-testid="stFileUploader"]:hover {
        /* Hover effect removed */
    }

    [data-testid="stFileUploader"] label {
        color: var(--text-secondary) !important;
    }
    </style>
    """, unsafe_allow_html=True)