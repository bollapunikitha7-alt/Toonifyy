import re

def validate_email(email: str) -> tuple[bool, str]:
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not email:
        return False, "Email is required."
    if not re.match(pattern, email):
        return False, "Enter a valid email address."
    return True, ""

def validate_password(password: str) -> tuple[bool, str]:
    if len(password) < 8:
        return False, "Password must be at least 8 characters."
    if not re.search(r'[A-Z]', password):
        return False, "Must contain at least one uppercase letter."
    if not re.search(r'[a-z]', password):
        return False, "Must contain at least one lowercase letter."
    if not re.search(r'\d', password):
        return False, "Must contain at least one number."
    if not re.search(r'[!@#$%^&*(),.?\":{}|<>]', password):
        return False, "Must contain at least one special character."
    return True, ""

def validate_username(username: str) -> tuple[bool, str]:
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(username) > 20:
        return False, "Username must be under 20 characters."
    if not re.match(r'^[a-zA-Z0-9_]+$', username):
        return False, "Username can only contain letters, numbers, and underscores."
    return True, ""