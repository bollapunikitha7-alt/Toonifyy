import streamlit as st
from auth.authenticator import register_user
from auth.google_oauth  import get_google_auth_url
from auth.validators    import validate_email, validate_password, validate_username

def _password_strength(pw: str) -> tuple[int, str, str]:
    score = 0
    if len(pw) >= 8: score += 1
    if any(c.isupper() for c in pw): score += 1
    if any(c.isdigit() for c in pw): score += 1
    if any(c in '!@#$%^&*' for c in pw): score += 1
    labels = ["", "Weak", "Fair", "Good", "Strong"]
    colors = ["", "#ef4444", "#f97316", "#eab308", "#22c55e"]
    return score, labels[score], colors[score]

def show_register():
    st.markdown("""
    <div class='auth-wrapper'>
        <div class='auth-header'>
            <h2>Create Account 🎨</h2>
            <p>Join thousands of artists using Toonify</p>
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
    
    username = st.text_input("Username", placeholder="Choose a username", key="reg_user")
    email = st.text_input("Email", placeholder="Enter your email address", key="reg_email")
    password = st.text_input("Password", placeholder="Create a password", type="password", key="reg_pass")
    
    if password:
        score, label, color = _password_strength(password)
        st.markdown(f"""
        <div class='password-strength'>
            <div class='strength-bars'>
                <div class='strength-bar {"active" if score >= 1 else ""}'></div>
                <div class='strength-bar {"active" if score >= 2 else ""}'></div>
                <div class='strength-bar {"active" if score >= 3 else ""}'></div>
                <div class='strength-bar {"active" if score >= 4 else ""}'></div>
            </div>
            <div class='strength-text' style='color: {color};'>{label} password</div>
        </div>
        """, unsafe_allow_html=True)
    
    confirm = st.text_input("Confirm Password", placeholder="Confirm your password", type="password", key="reg_confirm")
    
    # Create Account button
    if st.button("Create Account", key="btn_create", use_container_width=True):
        ok_u, msg_u = validate_username(username)
        ok_e, msg_e = validate_email(email)
        ok_p, msg_p = validate_password(password)
        
        if not ok_u:
            st.error(msg_u)
        elif not ok_e:
            st.error(msg_e)
        elif not ok_p:
            st.error(msg_p)
        elif password != confirm:
            st.error("Passwords do not match.")
        else:
            ok, result = register_user(username, email, password)
            if ok:
                st.success("✅ Account created successfully! Welcome to Toonify!")
                st.session_state.jwt_token = result
                st.session_state.authenticated = True
                st.session_state.current_user = {"username": username, "email": email}
                st.rerun()
            else:
                st.error(result)
    
    # Footer with navigation
    st.markdown("""
        <div class='auth-footer'>
            <p>Already have an account?</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Navigation buttons in a single row
    col1, col2 = st.columns(2)
    # with col1:
    #     if st.button("← Back to Home", key="back_to_home_register", use_container_width=True):
    #         st.session_state.show_auth = False
    #         st.rerun()
    with col2:
        if st.button("Sign In →", key="goto_login_from_register", use_container_width=True):
            st.session_state.auth_mode = "login"
            st.rerun()
    
    # Close the auth-body and auth-wrapper divs
    st.markdown("</div></div>", unsafe_allow_html=True)