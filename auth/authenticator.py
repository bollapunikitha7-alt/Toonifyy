import json, os, re, bcrypt, jwt, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
JWT_SECRET  = os.getenv("JWT_SECRET_KEY", "fallback-secret-key-change-in-prod")
USERS_FILE  = Path("data/users.json")

def _load_users() -> dict:
    USERS_FILE.parent.mkdir(exist_ok=True)
    if not USERS_FILE.exists():
        USERS_FILE.write_text("{}")
    return json.loads(USERS_FILE.read_text())

def _save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=2))

def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()

def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode(), hashed.encode())

def create_jwt(user_id: str, email: str) -> str:
    payload = {
        "sub": user_id,
        "email": email,
        "iat": int(time.time()),
        "exp": int(time.time()) + 60 * 60 * 8,  # 8 hour expiry
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def verify_jwt(token: str) -> dict | None:
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def register_user(username: str, email: str, password: str) -> tuple[bool, str]:
    users = _load_users()
    for u in users.values():
        if u["email"] == email:
            return False, "An account with this email already exists."
    if username in users:
        return False, "Username already taken."
    users[username] = {
        "username": username,
        "email":    email,
        "password": hash_password(password),
        "provider": "local",
        "created_at": int(time.time()),
    }
    _save_users(users)
    return True, create_jwt(username, email)

def login_user(email: str, password: str) -> tuple[bool, str, dict]:
    users = _load_users()
    for username, u in users.items():
        if u["email"] == email and u.get("provider", "local") == "local":
            if verify_password(password, u["password"]):
                token = create_jwt(username, email)
                return True, token, {"username": username, "email": email}
    return False, "Invalid email or password.", {}

def upsert_google_user(google_info: dict) -> tuple[bool, str, dict]:
    """Create or update a Google OAuth user."""
    users   = _load_users()
    email   = google_info["email"]
    name    = google_info.get("name", email.split("@")[0])
    user_id = f"google_{google_info['sub']}"

    if user_id not in users:
        users[user_id] = {
            "username":   name,
            "email":      email,
            "password":   "",
            "provider":   "google",
            "picture":    google_info.get("picture", ""),
            "created_at": int(time.time()),
        }
        _save_users(users)

    token = create_jwt(user_id, email)
    return True, token, {"username": name, "email": email,
                         "picture": users[user_id].get("picture", "")}
def _load_users() -> dict:
    USERS_FILE.parent.mkdir(exist_ok=True)
    if not USERS_FILE.exists():
        USERS_FILE.write_text("{}")
        return {}
    content = USERS_FILE.read_text().strip()
    if not content:
        USERS_FILE.write_text("{}")
        return {}
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        USERS_FILE.write_text("{}")
        return {}