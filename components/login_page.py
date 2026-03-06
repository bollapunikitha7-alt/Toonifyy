import streamlit as st
from auth.authenticator import login_user
from auth.google_oauth  import get_google_auth_url
from auth.validators    import validate_email

def show_login():
    st.markdown("""
    <div class='auth-wrapper'>
        <div class='auth-header'>
            <h2>Welcome Back! 👋</h2>
            <p>Sign in to continue your creative journey</p>
        </div>
        <div class='auth-body'>
    """, unsafe_allow_html=True)
    
    google_url = get_google_auth_url()
    st.markdown(f"""
        <a class='google-btn' href='{google_url}' target='_self'>
            <svg width='20' height='20' viewBox='0 0 18 18'>
                <path fill='#4285F4' d='M17.64 9.2c0-.637-.057-1.251-.164-1.84H9v3.481h4.844c-.209 1.125-.843 2.078-1.796 2.717v2.258h2.908c1.702-1.567 2.684-3.875 2.684-6.615z'/>
                <path fill='#34A853' d='M9 18c2.43 0 4.467-.806 5.956-2.18l-2.908-2.259c-.806.54-1.837.86-3.048.86-2.344 0-4.328-1.584-5.036-3.711H.957v2.332A8.997 8.997 0 0 0 9 18z'/>
                <path fill='#FBBC05' d='M3.964 10.71A5.41 5.41 0 0 1 3.682 9c0-.593.102-1.17.282-1.71V4.958H.957A8.996 8.996 0 0 0 0 9c0 1.452.348 2.827.957 4.042l3.007-2.332z'/>
                <path fill='#EA4335' d='M9 3.58c1.321 0 2.508.454 3.44 1.345l2.582-2.58C13.463.891 11.426 0 9 0A8.997 8.997 0 0 0 .957 4.958L3.964 7.29C4.672 5.163 6.656 3.58 9 3.58z'/>
            </svg>
            Continue with Google
        </a>
        
        <div class='auth-divider'>
            <div class='auth-divider-line'></div>
            <span class='auth-divider-text'>or</span>
            <div class='auth-divider-line'></div>
        </div>
    """, unsafe_allow_html=True)
    
    email = st.text_input("Email", placeholder="Enter your email address", key="login_email")
    password = st.text_input("Password", placeholder="Enter your password", type="password", key="login_pass")
    
    # Sign In button
    if st.button("Sign In", key="btn_signin", use_container_width=True):
        ok_e, msg_e = validate_email(email)
        if not ok_e:
            st.error(msg_e)
        elif not password:
            st.error("Password is required.")
        else:
            ok, token, user = login_user(email, password)
            if ok:
                st.session_state.jwt_token = token
                st.session_state.authenticated = True
                st.session_state.current_user = user
                st.rerun()
            else:
                st.error(token)
    
    # Footer with navigation
    st.markdown("""
        <div class='auth-footer'>
            <p>Don't have an account?</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons in a single row
    col1, col2 = st.columns(2)
    # with col1:
    #     if st.button("← Back to Home", key="back_to_home_login", use_container_width=True):
    #         st.session_state.show_auth = False
    #         st.rerun()
    with col2:
        if st.button("Create Account →", key="goto_register_from_login", use_container_width=True):
            st.session_state.auth_mode = "register"
            st.rerun()
    
    # Close the auth-body and auth-wrapper divs
    st.markdown("</div></div>", unsafe_allow_html=True)