# app.py
import streamlit as st

from utils.styles         import inject_global_css
from auth.authenticator   import verify_jwt, upsert_google_user
from auth.google_oauth    import exchange_code_for_token, get_google_user_info
from components.login_page    import show_login
from components.register_page import show_register
from components.editor        import show_editor, PRESET_FILTERS  # Import PRESET_FILTERS

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="TOONIFY", layout="wide", page_icon="🎨")
inject_global_css()

# ── Session defaults ──────────────────────────────────────────────────────────
# Get the first preset key from PRESET_FILTERS (which includes emoji)
first_preset = list(PRESET_FILTERS.keys())[0] if 'PRESET_FILTERS' in dir() else "🎨 Ghibli Soft"

for k, v in {
    "authenticated": False,
    "jwt_token":     None,
    "current_user":  None,
    "auth_mode":     "register",        # ← register opens first
    "uploaded_image": None,
    "active_preset": first_preset,      # ← Use the emoji version
    "view_mode":     "Split",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Handle Google OAuth callback ──────────────────────────────────────────────
params = st.query_params
if "code" in params and not st.session_state.authenticated:
    code       = params["code"]
    token_data = exchange_code_for_token(code)
    if token_data and "access_token" in token_data:
        user_info = get_google_user_info(token_data["access_token"])
        if user_info:
            ok, jwt_tok, user = upsert_google_user(user_info)
            if ok:
                st.session_state.jwt_token     = jwt_tok
                st.session_state.authenticated = True
                st.session_state.current_user  = user
                st.query_params.clear()
                st.rerun()
    st.error("Google sign-in failed. Please try again.")

# ── JWT session validation ────────────────────────────────────────────────────
if st.session_state.jwt_token and not st.session_state.authenticated:
    payload = verify_jwt(st.session_state.jwt_token)
    if payload:
        st.session_state.authenticated = True

# ── Router ────────────────────────────────────────────────────────────────────
if st.session_state.authenticated:
    show_editor()
else:
    _, col, _ = st.columns([1.2, 1, 1.2])   # ← narrower center column = smaller forms
    with col:
        if st.session_state.auth_mode == "login":
            show_login()
        else:
            show_register()