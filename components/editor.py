# editor.py — TOONIFY Streamlit App
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from datetime import datetime
import hashlib
import json
import os
import time
from pathlib import Path

# Import new modules
from components.download_prep import DownloadPreparation
from components.payment_handler import payment_handler

# ── Authentication Functions ─────────────────────────────────────────────────

def hash_password(password):
    """Hash a password for storing."""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users():
    """Load users from JSON file"""
    users_file = Path(__file__).parent / "data" / "users.json"
    if users_file.exists():
        with open(users_file, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    """Save users to JSON file"""
    users_file = Path(__file__).parent / "data" / "users.json"
    users_file.parent.mkdir(parents=True, exist_ok=True)
    with open(users_file, 'w') as f:
        json.dump(users, f, indent=2)

def register_user(username, email, password, name=None):
    """Register a new user"""
    users = load_users()
    
    # Check if username already exists
    if username in users:
        return False, "Username already exists"
    
    # Check if email already exists
    for user_data in users.values():
        if user_data.get('email') == email:
            return False, "Email already registered"
    
    # Create new user
    users[username] = {
        'username': username,
        'email': email,
        'password': hash_password(password),
        'name': name or username,
        'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'downloads': []
    }
    
    save_users(users)
    return True, "Registration successful!"

def login_user(username, password):
    """Login a user"""
    users = load_users()
    
    if username not in users:
        return False, "Username not found"
    
    if users[username]['password'] != hash_password(password):
        return False, "Incorrect password"
    
    return True, users[username]

# ── Filter Functions ──────────────────────────────────────────────────────────
def to_cv(pil_img):
    return cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)

def to_pil(cv_img):
    return Image.fromarray(cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB))

def apply_ghibli_soft(img, line_w, smooth, detail, color_simp):
    cv_img = to_cv(img)
    d = int(9 + smooth * 6)
    d = d if d % 2 == 1 else d + 1
    smooth_img = cv2.bilateralFilter(cv_img, d=d, sigmaColor=75 + smooth * 50, sigmaSpace=75 + smooth * 50)
    hsv = cv2.cvtColor(smooth_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + (1 - color_simp) * 0.6), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(
        cv2.GaussianBlur(gray, (5, 5), 0), 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
        blockSize=max(3, int(9 - line_w * 4)) | 1, C=int(2 + line_w * 6)
    )
    return to_pil(cv2.bitwise_and(result, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))

def apply_cell_shade(img, line_w, smooth, detail, color_simp):
    cv_img = to_cv(img)
    k = max(2, int(4 + color_simp * 8))
    data = cv_img.reshape((-1, 3)).astype(np.float32)
    _, labels, centers = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 8, cv2.KMEANS_RANDOM_CENTERS)
    quantized = centers[labels.flatten()].reshape(cv_img.shape).astype(np.uint8)
    gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=max(3, int(11 - line_w * 4)) | 1, C=int(3 + line_w * 5))
    return to_pil(cv2.bitwise_and(quantized, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))

def apply_bw_ink(img, line_w, smooth, detail, color_simp):
    cv_img = to_cv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    k = max(1, int(smooth * 4)) | 1
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=max(3, int(13 - line_w * 6)) | 1, C=int(4 + line_w * 8))
    detail_layer = cv2.Canny(blur, int(30 + detail * 50), int(80 + detail * 100))
    result = cv2.bitwise_and(edges, cv2.bitwise_not(detail_layer))
    return to_pil(cv2.cvtColor(result, cv2.COLOR_GRAY2BGR))

def apply_vector_flat(img, line_w, smooth, detail, color_simp):
    cv_img = to_cv(img)
    flat = cv2.pyrMeanShiftFiltering(cv_img, sp=int(10 + smooth * 15), sr=int(20 + color_simp * 40))
    k = max(3, int(5 + color_simp * 10))
    data = flat.reshape((-1, 3)).astype(np.float32)
    _, labels, centers = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5), 6, cv2.KMEANS_RANDOM_CENTERS)
    quantized = centers[labels.flatten()].reshape(flat.shape).astype(np.uint8)
    gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, int(20 + line_w * 30), int(60 + line_w * 80))
    kernel = np.ones((max(1, int(line_w * 2)), max(1, int(line_w * 2))), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges_inv = cv2.bitwise_not(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    return to_pil(cv2.bitwise_and(quantized, edges_inv))

def apply_pencil_sketch(img, line_w, smooth, detail, color_simp):
    cv_img = to_cv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blur_size = int(15 + smooth * 20) | 1
    blurred = cv2.GaussianBlur(inverted, (blur_size, blur_size), 0)
    sketch = cv2.divide(gray, 255 - blurred, scale=256)
    sketch = cv2.equalizeHist(sketch)
    if detail > 0.3:
        edges = cv2.Canny(gray, 50, 150)
        sketch = cv2.addWeighted(sketch, 1, edges, detail * 0.3, 0)
    return to_pil(cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR))

def apply_watercolor(img, line_w, smooth, detail, color_simp):
    cv_img = to_cv(img)
    for _ in range(3):
        cv_img = cv2.bilateralFilter(cv_img, d=9, sigmaColor=75, sigmaSpace=75)
    k = max(8, int(12 - color_simp * 8))
    data = cv_img.reshape((-1, 3)).astype(np.float32)
    _, labels, centers = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.5), 5, cv2.KMEANS_RANDOM_CENTERS)
    quantized = centers[labels.flatten()].reshape(cv_img.shape).astype(np.uint8)
    gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
    edges_inv = cv2.bitwise_not(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    result = cv2.bitwise_and(quantized, edges_inv)
    return to_pil(cv2.GaussianBlur(result, (3, 3), 0))

def apply_comic_book(img, line_w, smooth, detail, color_simp):
    cv_img = to_cv(img)
    lab = cv2.cvtColor(cv_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    k = max(6, int(10 - color_simp * 6))
    data = enhanced.reshape((-1, 3)).astype(np.float32)
    _, labels, centers = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 15, 0.5), 6, cv2.KMEANS_RANDOM_CENTERS)
    quantized = centers[labels.flatten()].reshape(enhanced.shape).astype(np.uint8)
    gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=max(3, int(9 - line_w * 4)) | 1, C=2)
    kernel = np.ones((max(1, int(line_w * 2)), max(1, int(line_w * 2))), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    return to_pil(cv2.bitwise_and(quantized, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))

def apply_charcoal(img, line_w, smooth, detail, color_simp):
    cv_img = to_cv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur_size = int(5 + smooth * 10) | 1
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    sketch = cv2.divide(gray, blurred, scale=256)
    noise = np.random.normal(0, 25 * smooth, sketch.shape).astype(np.uint8)
    sketch = cv2.add(sketch, noise)
    edges = cv2.Canny(gray, 30, 100)
    sketch = cv2.addWeighted(sketch, 1, edges, 0.3, 0)
    return to_pil(cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR))

def apply_pop_art(img, line_w, smooth, detail, color_simp):
    cv_img = to_cv(img)
    hsv = cv2.cvtColor(cv_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.8, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255)
    saturated = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    k = max(4, int(8 - color_simp * 4))
    data = saturated.reshape((-1, 3)).astype(np.float32)
    _, labels, centers = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.5), 5, cv2.KMEANS_RANDOM_CENTERS)
    quantized = centers[labels.flatten()].reshape(saturated.shape).astype(np.uint8)
    gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 60)
    edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=2)
    edges_3ch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return to_pil(cv2.bitwise_and(quantized, cv2.bitwise_not(edges_3ch)))

def apply_neon_glow(img, line_w, smooth, detail, color_simp):
    cv_img = to_cv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 100)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=int(1 + line_w * 3))
    glow = np.zeros_like(cv_img)
    for i in range(3):
        glow[:, :, i] = cv2.bitwise_and(cv_img[:, :, i], edges)
    glow = cv2.GaussianBlur(glow, (15, 15), 0)
    result = cv2.addWeighted(cv_img, 0.3, glow, 0.7, 0)
    return to_pil(result)

def apply_vintage(img, line_w, smooth, detail, color_simp):
    cv_img = to_cv(img)
    sepia = np.array(to_pil(cv_img))
    sepia = cv2.transform(sepia, np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]]))
    sepia = np.clip(sepia, 0, 255).astype(np.uint8)
    grain = np.random.normal(0, 15, sepia.shape).astype(np.uint8)
    sepia = cv2.add(sepia, grain)
    h, w = sepia.shape[:2]
    kernel_x = cv2.getGaussianKernel(w, w / 3)
    kernel_y = cv2.getGaussianKernel(h, h / 3)
    mask = (kernel_y * kernel_x.T)
    mask = np.stack([mask / mask.max()] * 3, axis=2)
    return to_pil(cv2.cvtColor((sepia * mask).astype(np.uint8), cv2.COLOR_RGB2BGR))

def apply_mosaic(img, line_w, smooth, detail, color_simp):
    cv_img = to_cv(img)
    pixel_size = max(4, int(20 - smooth * 18))
    h, w = cv_img.shape[:2]
    small = cv2.resize(cv_img, (w // pixel_size, h // pixel_size), interpolation=cv2.INTER_LINEAR)
    return to_pil(cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST))

def apply_thermal(img, line_w, smooth, detail, color_simp):
    cv_img = to_cv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    cmap = cv2.COLORMAP_HOT if color_simp > 0.5 else cv2.COLORMAP_JET
    return to_pil(cv2.applyColorMap(gray, cmap))

def apply_pastel(img, line_w, smooth, detail, color_simp):
    cv_img = to_cv(img)
    smooth_img = cv2.bilateralFilter(cv_img, d=9, sigmaColor=100, sigmaSpace=100)
    hsv = cv2.cvtColor(smooth_img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 0.7, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.1, 0, 255)
    result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    texture = np.random.normal(0, 5, result.shape).astype(np.uint8)
    return to_pil(cv2.addWeighted(result, 0.95, texture, 0.05, 0))

def apply_stained_glass(img, line_w, smooth, detail, color_simp):
    cv_img = to_cv(img)
    filtered = cv2.pyrMeanShiftFiltering(cv_img, sp=20, sr=30)
    k = max(6, int(12 - color_simp * 8))
    data = filtered.reshape((-1, 3)).astype(np.float32)
    _, labels, centers = cv2.kmeans(data, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.5), 5, cv2.KMEANS_RANDOM_CENTERS)
    quantized = centers[labels.flatten()].reshape(filtered.shape).astype(np.uint8)
    gray = cv2.cvtColor(quantized, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=int(2 + line_w * 3))
    edges_3ch = cv2.bitwise_not(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))
    return to_pil(cv2.bitwise_and(quantized, edges_3ch))

def apply_ink_wash(img, line_w, smooth, detail, color_simp):
    cv_img = to_cv(img)
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    blur_size = int(15 + smooth * 20) | 1
    blurred = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)
    wash = cv2.divide(gray, blurred, scale=200)
    wash = cv2.equalizeHist(wash)
    wash = cv2.addWeighted(wash, 0.8, np.zeros_like(wash), 0, 20)
    return to_pil(cv2.cvtColor(wash, cv2.COLOR_GRAY2BGR))

# ── Registry ──────────────────────────────────────────────────────────────────

PRESET_FILTERS = {
    "🎨 Ghibli Soft": apply_ghibli_soft,
    "⚡ Cell Shade": apply_cell_shade,
    "🖋️ B&W Ink": apply_bw_ink,
    "📐 Vector Flat": apply_vector_flat,
    "✏️ Pencil Sketch": apply_pencil_sketch,
    "💧 Watercolor": apply_watercolor,
    "📖 Comic Book": apply_comic_book,
    "🔥 Charcoal": apply_charcoal,
    "🎭 Pop Art": apply_pop_art,
    "💡 Neon Glow": apply_neon_glow,
    "📷 Vintage": apply_vintage,
    "🧩 Mosaic": apply_mosaic,
    "🌡️ Thermal": apply_thermal,
    "🌸 Pastel": apply_pastel,
    "🌈 Stained Glass": apply_stained_glass,
    "🀄 Ink Wash": apply_ink_wash,
}

FILTER_CATEGORIES = {
    "🖼️ Classic": ["🎨 Ghibli Soft", "⚡ Cell Shade", "🖋️ B&W Ink", "📐 Vector Flat"],
    "✏️ Sketch": ["✏️ Pencil Sketch", "🔥 Charcoal", "🀄 Ink Wash"],
    "🎨 Artistic": ["💧 Watercolor", "🌸 Pastel", "🎭 Pop Art"],
    "🎯 Effects": ["📖 Comic Book", "💡 Neon Glow", "🌡️ Thermal", "🧩 Mosaic", "📷 Vintage", "🌈 Stained Glass"],
}

# ── Payment and Download UI Functions ─────────────────────────────────────────

def show_payment_page(order_id: str, amount: int, image_path: str, style_name: str, format: str):
    """Display payment page with Razorpay integration"""
    
    st.markdown("""
    <style>
    .payment-container {
        max-width: 600px;
        margin: 2rem auto;
        padding: 2rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .price-tag {
        font-size: 2.5rem;
        font-weight: 700;
        color: #3B82F6;
        text-align: center;
        margin: 1rem 0;
    }
    .feature-list {
        list-style: none;
        padding: 0;
        margin: 2rem 0;
    }
    .feature-list li {
        padding: 0.5rem 0;
        border-bottom: 1px solid #eee;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .feature-list li:before {
        content: "✓";
        color: #10B981;
        font-weight: bold;
        margin-right: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="payment-container">', unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <h2 style="text-align: center; color: #333; margin-bottom: 2rem;">
        💳 Complete Your Purchase
    </h2>
    """, unsafe_allow_html=True)
    
    # Preview image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if os.path.exists(image_path):
            img = Image.open(image_path)
            st.image(img, use_container_width=True, caption="Your Styled Image")
    
    # Order summary
    st.markdown("### 📋 Order Summary")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Style:** {style_name}")
        st.markdown(f"**Format:** {format}")
    with col2:
        st.markdown(f"**Quality:** High (lossless)")
        st.markdown(f"**Watermark:** Removed after payment")
    
    # Price
    st.markdown(f"""
    <div class="price-tag">
        ₹{amount/100:.2f}
    </div>
    """, unsafe_allow_html=True)
    
    # Features
    st.markdown("""
    <ul class="feature-list">
        <li>High-quality, watermark-free image</li>
        <li>Multiple format support (PNG, JPG, WEBP)</li>
        <li>30-day re-download access</li>
        <li>Secure payment processing</li>
        <li>Instant download after payment</li>
    </ul>
    """, unsafe_allow_html=True)
    
    # Payment button
    st.markdown("### 💳 Payment")
    
    # Create Razorpay order if not exists
    if 'razorpay_order' not in st.session_state:
        order = payment_handler.create_payment_order(
            amount=amount,
            notes={
                'style': style_name,
                'format': format,
                'user_id': st.session_state.current_user.get('username', 'guest')
            }
        )
        if order:
            st.session_state.razorpay_order = order
            st.session_state.payment_order_id = order['id']
    
    # Payment button (simulated for demo)
    if st.button("💳 Pay Now", use_container_width=True, type="primary"):
        with st.spinner("Processing payment..."):
            # Simulate payment processing
            time.sleep(2)
            
            # Simulate successful payment
            payment_response = {
                'razorpay_order_id': st.session_state.payment_order_id,
                'razorpay_payment_id': f'pay_{int(time.time())}',
                'razorpay_signature': 'simulated_signature'
            }
            
            # Verify payment
            if payment_handler.verify_payment_signature(payment_response):
                st.session_state.payment_successful = True
                st.session_state.payment_details = payment_response
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Security badges
    st.markdown("""
    <div style="text-align: center; margin-top: 2rem; color: #666; font-size: 0.9rem;">
        🔒 Secure payment powered by Razorpay<br>
        Your payment information is encrypted and secure
    </div>
    """, unsafe_allow_html=True)

def show_payment_success_page():
    """Display payment success page with download options"""
    
    st.markdown("""
    <style>
    .success-container {
        max-width: 600px;
        margin: 2rem auto;
        padding: 2rem;
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(16, 185, 129, 0.3);
    }
    .success-icon {
        font-size: 4rem;
        margin-bottom: 1rem;
    }
    .transaction-id {
        background: rgba(255,255,255,0.2);
        padding: 0.5rem;
        border-radius: 8px;
        font-family: monospace;
        margin: 1rem 0;
    }
    .download-section {
        max-width: 600px;
        margin: 2rem auto;
        padding: 2rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Success message
    st.markdown(f"""
    <div class="success-container">
        <div class="success-icon">✅</div>
        <h2>Payment Successful!</h2>
        <p>Thank you for your purchase. Your transaction has been completed successfully.</p>
        <div class="transaction-id">
            Transaction ID: {st.session_state.payment_details.get('razorpay_payment_id', 'N/A')}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Download options
    st.markdown('<div class="download-section">', unsafe_allow_html=True)
    st.markdown("### 📥 Download Your Image")
    
    download_info = st.session_state.get('download_info', {})
    if download_info and os.path.exists(download_info['path']):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # PNG Download (high quality)
            file_bytes = DownloadPreparation.get_download_bytes(download_info['path'])
            if file_bytes:
                st.download_button(
                    label="📷 PNG (Lossless)",
                    data=file_bytes,
                    file_name=download_info['filename'],
                    mime="image/png",
                    use_container_width=True
                )
        
        with col2:
            # JPG Download
            img = Image.open(download_info['path'])
            img_jpg = img.convert('RGB')
            buf = io.BytesIO()
            img_jpg.save(buf, format="JPEG", quality=95)
            st.download_button(
                label="🖼️ JPG (Compressed)",
                data=buf.getvalue(),
                file_name=download_info['filename'].replace('.png', '.jpg').replace('.webp', '.jpg'),
                mime="image/jpeg",
                use_container_width=True
            )
        
        with col3:
            # WEBP Download
            img = Image.open(download_info['path'])
            buf = io.BytesIO()
            img.save(buf, format="WEBP", quality=95)
            st.download_button(
                label="🌐 WEBP (Modern)",
                data=buf.getvalue(),
                file_name=download_info['filename'].replace('.png', '.webp').replace('.jpg', '.webp'),
                mime="image/webp",
                use_container_width=True
            )
        
        st.markdown("---")
        
        # Comparison download
        if st.button("🔄 Download Original vs Styled Comparison", use_container_width=True):
            if 'uploaded_image' in st.session_state:
                original = st.session_state.uploaded_image
                styled = Image.open(download_info['path'])
                
                # Create comparison image
                comparison = Image.new('RGB', (original.width * 2, original.height))
                comparison.paste(original, (0, 0))
                comparison.paste(styled, (original.width, 0))
                
                buf = io.BytesIO()
                comparison.save(buf, format="PNG")
                st.download_button(
                    label="⬇ Download Comparison",
                    data=buf.getvalue(),
                    file_name=f"comparison_{download_info['filename']}",
                    mime="image/png",
                    key="download_comparison"
                )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Download history
    st.markdown("---")
    st.markdown("### 📚 Your Downloads")
    st.info("Your purchased images will be available for re-download for 30 days.")
    
    # Navigation
    if st.button("← Back to Editor", use_container_width=True):
        st.session_state.payment_page = False
        st.session_state.payment_successful = False
        st.session_state.razorpay_order = None
        st.rerun()

def show_payment_failure_page():
    """Display payment failure page"""
    
    st.markdown("""
    <style>
    .failure-container {
        max-width: 600px;
        margin: 2rem auto;
        padding: 2rem;
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        border-radius: 20px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(239, 68, 68, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="failure-container">
        <div style="font-size: 4rem; margin-bottom: 1rem;">❌</div>
        <h2>Payment Failed</h2>
        <p>We couldn't process your payment. Please try again or contact support.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔄 Retry Payment", use_container_width=True):
            st.session_state.payment_page = True
            st.session_state.payment_successful = False
            st.rerun()
    
    with col2:
        if st.button("📧 Contact Support", use_container_width=True):
            st.info("Please email: support@toonify.com")
    
    if st.button("← Back to Editor", use_container_width=True):
        st.session_state.payment_page = False
        st.rerun()

def show_download_history():
    """Show user's download history"""
    st.markdown("### 📚 Download History")
    
    user = st.session_state.current_user
    if user and 'downloads' in user and user['downloads']:
        for download in user['downloads'][-5:]:  # Show last 5 downloads
            with st.container():
                col1, col2, col3 = st.columns([3, 2, 2])
                with col1:
                    st.markdown(f"**{download['style']}**")
                    st.caption(download['date'])
                with col2:
                    st.markdown(f"Format: {download['format']}")
                with col3:
                    if st.button(f"📥 Re-download", key=f"redl_{download['id']}"):
                        st.info("Re-download feature coming soon!")
                st.markdown("---")
    else:
        st.info("No download history yet. Purchase your first styled image to see it here!")

# ── Authentication UI ─────────────────────────────────────────────────────────

def show_auth_forms():
    """Display login and registration forms in the dashboard"""
    st.markdown("""
    <style>
    .auth-container {
        max-width: 500px;
        margin: 2rem auto;
        padding: 2rem;
        background: white;
        border-radius: 20px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .auth-title {
        text-align: center;
        color: #333;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin: 2rem 0;">
        <h1 style="font-size: 3rem; color: #333;">🎨 TOONIFY</h1>
        <p style="font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
            Transform your photos into amazing artworks!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="auth-container">', unsafe_allow_html=True)
    st.markdown('<h2 class="auth-title">🔐 Authentication Required</h2>', unsafe_allow_html=True)
    
    # Create tabs for login/register
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
    
    with tab1:
        with st.form("login_form"):
            st.markdown("#### Login to Your Account")
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            if st.form_submit_button("Login", use_container_width=True, type="primary"):
                if username and password:
                    success, result = login_user(username, password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.current_user = result
                        st.success("Login successful! Welcome back!")
                        st.rerun()
                    else:
                        st.error(result)
                else:
                    st.warning("Please fill in all fields")
    
    with tab2:
        with st.form("register_form"):
            st.markdown("#### Create New Account")
            reg_username = st.text_input("Username", placeholder="Choose a username", key="reg_username")
            reg_email = st.text_input("Email", placeholder="Enter your email", key="reg_email")
            reg_name = st.text_input("Full Name (optional)", placeholder="Enter your full name", key="reg_name")
            reg_password = st.text_input("Password", type="password", placeholder="Choose a password", key="reg_password")
            reg_confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password", key="reg_confirm")
            
            if st.form_submit_button("Register", use_container_width=True, type="primary"):
                if not all([reg_username, reg_email, reg_password, reg_confirm_password]):
                    st.warning("Please fill in all required fields")
                elif reg_password != reg_confirm_password:
                    st.error("Passwords do not match")
                else:
                    success, message = register_user(reg_username, reg_email, reg_password, reg_name)
                    if success:
                        st.success(message)
                        st.session_state.auth_tab = "login"
                        st.rerun()
                    else:
                        st.error(message)
    
    st.markdown('</div>', unsafe_allow_html=True)

# ── Dashboard Welcome Component ─────────────────────────────────────────────────

def show_dashboard_welcome():
    """Display dashboard welcome message with user info"""
    user = st.session_state.get('current_user', {})
    
    # Handle case when user is None or empty
    if not user:
        username = "User"
        email = ""
    else:
        username = user.get('name', user.get('username', 'User')).split()[0] if user.get('name') else user.get('username', 'User').split()[0]
        email = user.get('email', '')
    
    # Get stats from session state or use defaults
    filters_applied = len(st.session_state.get('recent_filters', []))
    
    # Welcome header with gradient background
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 20px;
        padding: 2rem;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    ">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div>
                <h1 style="margin:0; font-size:2.5rem;">Welcome back, {username}! 👋</h1>
                <p style="margin:0.5rem 0 0 0; opacity:0.9;">{email}</p>
            </div>
            <div style="
                background: rgba(255,255,255,0.2);
                border-radius: 15px;
                padding: 1rem;
                text-align: center;
                min-width: 120px;
            ">
                <div style="font-size: 2rem; font-weight: 700;">{filters_applied}</div>
                <div style="opacity: 0.9;">Filters Applied</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def show_user_info():
    """Display user info in a compact format next to upload"""
    user = st.session_state.get('current_user', {})
    
    # Handle case when user is None or empty
    if not user:
        username = "User"
        avatar_initial = "U"
    else:
        username = user.get('name', user.get('username', 'User'))
        avatar_initial = username[0].upper() if username else 'U'
    
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 0.5rem;">
        <div style="
            width: 32px;
            height: 32px;
            background: linear-gradient(135deg, #3B82F6 0%, #60A5FA 100%);
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            font-weight: 600;
            color: white;
            box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
        ">{avatar_initial}</div>
        <span style="color: #666; font-size: 0.9rem; font-weight: 500;">{username}</span>
    </div>
    """, unsafe_allow_html=True)

def show_settings_menu():
    """Display settings menu in a popover"""
    with st.popover("⚙️", use_container_width=True):
        if st.session_state.get('authenticated', False):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 👤 Account")
                if st.button("📋 Profile", use_container_width=True):
                    st.session_state.show_profile_settings = True
                
                if st.button("📚 Downloads", use_container_width=True):
                    st.session_state.show_download_history = True
            
            with col2:
                st.markdown("##### ⚙️ Preferences")
                if st.button("🎨 Theme", use_container_width=True):
                    st.session_state.show_theme = True
                
                if st.button("💳 Payment Methods", use_container_width=True):
                    st.info("Payment methods coming soon!")
            
            st.markdown("---")
        
        # Show logout when authenticated
        if st.session_state.get('authenticated', False):
            if st.button("🚪 Logout", use_container_width=True, type="primary"):
                st.session_state.authenticated = False
                st.session_state.current_user = None
                st.session_state.uploaded_image = None
                st.session_state.recent_filters = []
                st.session_state.payment_page = False
                st.session_state.payment_successful = False
                st.success("Logged out successfully!")
                st.rerun()

def show_profile_modal():
    """Display profile settings modal"""
    if st.session_state.get('show_profile_settings', False):
        with st.container():
            st.markdown("""
            <style>
            .modal-overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.5);
                z-index: 999;
            }
            .modal-content {
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: white;
                border-radius: 24px;
                padding: 2rem;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                z-index: 1000;
                width: 400px;
                max-width: 90%;
            }
            </style>
            <div class="modal-overlay"></div>
            <div class="modal-content">
            """, unsafe_allow_html=True)
            
            st.markdown("#### ✏️ Edit Profile")
            
            user = st.session_state.get('current_user', {})
            
            with st.form("profile_form"):
                name = st.text_input("Name", value=user.get('name', '') if user else '')
                email = st.text_input("Email", value=user.get('email', '') if user else '', disabled=True)
                bio = st.text_area("Bio", placeholder="Tell us about yourself...")
                
                col1, col2 = st.columns(2)
                with col1:
                    if st.form_submit_button("💾 Save"):
                        if 'current_user' in st.session_state and st.session_state.current_user:
                            st.session_state.current_user['name'] = name
                            st.session_state.current_user['bio'] = bio
                        st.success("Profile updated!")
                        st.session_state.show_profile_settings = False
                        st.rerun()
                
                with col2:
                    if st.form_submit_button("❌ Cancel"):
                        st.session_state.show_profile_settings = False
                        st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)

def show_theme_settings():
    """Display theme settings modal"""
    if st.session_state.get('show_theme', False):
        with st.container():
            st.markdown("""
            <style>
            .modal-overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.5);
                z-index: 999;
            }
            .modal-content {
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: white;
                border-radius: 24px;
                padding: 2rem;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                z-index: 1000;
                width: 400px;
                max-width: 90%;
            }
            </style>
            <div class="modal-overlay"></div>
            <div class="modal-content">
            """, unsafe_allow_html=True)
            
            st.markdown("#### 🎨 Theme Settings")
            
            theme = st.selectbox("Theme", ["Dark (Default)", "Light", "System"], index=0)
            accent_color = st.color_picker("Accent Color", "#3B82F6")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Apply", use_container_width=True):
                    st.success("Theme updated!")
                    st.session_state.show_theme = False
                    st.rerun()
            
            with col2:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.show_theme = False
                    st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)

def show_locked_message():
    """Show message that features are locked"""
    st.markdown("""
    <div style="
        height:400px;
        border:2px dashed #e0e0e0;
        border-radius:20px;
        display:flex;
        flex-direction: column;
        align-items:center;
        justify-content:center;
        font-size:1.2rem;
        color:#666;
        background:#f9f9f9;
        gap: 1rem;
    ">
        <div style="font-size: 4rem;">🔒</div>
        <div style="font-size: 1.8rem; font-weight: 600;">Features Locked</div>
        <div style="text-align: center; max-width: 400px;">
            Please login or register above to access the image editor and start creating amazing artworks!
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Main App ──────────────────────────────────────────────────────────────────

def show_editor():
    st.set_page_config(page_title="TOONIFY", layout="wide", page_icon="🎨")

    # Initialize session state
    if "active_preset" not in st.session_state:
        st.session_state.active_preset = list(PRESET_FILTERS.keys())[0]
    if "view_mode" not in st.session_state:
        st.session_state.view_mode = "Split"
    if "show_profile_settings" not in st.session_state:
        st.session_state.show_profile_settings = False
    if "show_theme" not in st.session_state:
        st.session_state.show_theme = False
    if "show_download_history" not in st.session_state:
        st.session_state.show_download_history = False
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None
    if "recent_filters" not in st.session_state:
        st.session_state.recent_filters = []
    if "payment_page" not in st.session_state:
        st.session_state.payment_page = False
    if "payment_successful" not in st.session_state:
        st.session_state.payment_successful = False
    if "styled_image" not in st.session_state:
        st.session_state.styled_image = None

    # Show modals if active
    show_profile_modal()
    show_theme_settings()
    
    if st.session_state.get('show_download_history', False):
        with st.container():
            st.markdown("""
            <style>
            .modal-overlay {
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.5);
                z-index: 999;
            }
            .modal-content {
                position: fixed;
                top: 50%;
                left: 50%;
                transform: translate(-50%, -50%);
                background: white;
                border-radius: 24px;
                padding: 2rem;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                z-index: 1000;
                width: 500px;
                max-width: 90%;
                max-height: 80vh;
                overflow-y: auto;
            }
            </style>
            <div class="modal-overlay"></div>
            <div class="modal-content">
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📚 Download History")
            show_download_history()
            
            if st.button("❌ Close", use_container_width=True):
                st.session_state.show_download_history = False
                st.rerun()
            
            st.markdown("</div>", unsafe_allow_html=True)

    # Check if we're on payment page
    if st.session_state.get('payment_page', False):
        if st.session_state.get('payment_successful', False):
            show_payment_success_page()
        else:
            # Show payment page
            if 'download_info' in st.session_state:
                download_info = st.session_state.download_info
                amount = payment_handler.calculate_amount(1, st.session_state.get('download_quality', 'high'))
                show_payment_page(
                    order_id=st.session_state.get('payment_order_id', ''),
                    amount=amount,
                    image_path=download_info['path'],
                    style_name=download_info['style'],
                    format=download_info['format']
                )
        return

    # Fixed Top Bar (always visible)
    col1, col2, col3, col4 = st.columns([2, 3, 1, 1])
    
    with col1:
        st.markdown("<h1 style='margin:0; font-size:2rem;'>🎨 TOONIFY</h1>", unsafe_allow_html=True)
    
    with col2:
        if st.session_state.authenticated:
            st.session_state.view_mode = st.radio(
                "View Mode", 
                ["Split", "Single"], 
                horizontal=True, 
                label_visibility="collapsed"
            )
    
    with col3:
        # User info (only show if authenticated)
        if st.session_state.authenticated:
            show_user_info()
    
    with col4:
        # Settings menu
        show_settings_menu()

    # Main content area - Show based on authentication
    if not st.session_state.authenticated:
        # Show login/register forms prominently
        show_auth_forms()
        
        # Show locked message for the editor area
        st.markdown("### 🎨 Image Editor (Locked)")
        show_locked_message()
        
    else:
        # User is authenticated - show dashboard welcome and editor
        show_dashboard_welcome()

        # File uploader for authenticated users
        uploaded = st.file_uploader(
            "Choose an image...", 
            type=["jpg", "jpeg", "png", "webp"],
            key="file_uploader"
        )
        if uploaded:
            st.session_state.uploaded_image = Image.open(uploaded)
            st.session_state.recent_filters.append(st.session_state.active_preset)

        # Main content area with editor
        canvas_col, sidebar_col = st.columns([2.6, 1], gap="large")

        with sidebar_col:
            st.markdown("#### 🎨 Style Gallery")
            for cat, presets in FILTER_CATEGORIES.items():
                st.markdown(f"**{cat}**")
                cols = st.columns(2)
                for i, preset in enumerate(presets):
                    with cols[i % 2]:
                        active = st.session_state.active_preset == preset
                        if st.button(preset, key=f"btn_{preset}", use_container_width=True,
                                   type="primary" if active else "secondary"):
                            st.session_state.active_preset = preset
                            st.rerun()

            st.markdown("---")
            st.markdown("#### ⚙️ Fine-Tuning")
            
            line_w = st.slider("🖊 Line Weight", 0.0, 1.0, 0.4, 0.05)
            color_simp = st.slider("🎨 Color Simplification", 0.0, 1.0, 0.75, 0.05)
            detail = st.slider("🔍 Detail Preservation", 0.0, 1.0, 0.25, 0.05)
            smooth = st.slider("✨ Smoothness", 0.0, 1.0, 0.5, 0.05)

            if "uploaded_image" in st.session_state and st.session_state.uploaded_image:
                styled = PRESET_FILTERS[st.session_state.active_preset](
                    st.session_state.uploaded_image, line_w, smooth, detail, color_simp
                )
                st.session_state.styled_image = styled
                
                st.markdown("---")
                st.markdown("#### 💾 Download Options")
                
                # Free preview download (with watermark)
                preview_path, preview_filename = DownloadPreparation.prepare_download(
                    image=styled,
                    user_id=st.session_state.current_user.get('username', 'guest'),
                    original_filename='preview.png',
                    style_name=st.session_state.active_preset,
                    format='PNG',
                    quality='medium',
                    add_watermark=True
                )
                
                if preview_path:
                    with open(preview_path, 'rb') as f:
                        preview_bytes = f.read()
                    
                    st.download_button(
                        "⬇ Free Preview (Watermarked)",
                        preview_bytes,
                        file_name=preview_filename,
                        mime="image/png",
                        use_container_width=True
                    )
                
                st.markdown("---")
                st.markdown("### 💎 Premium Download")
                st.markdown("*No watermark, high quality*")
                
                col1, col2 = st.columns(2)
                with col1:
                    format_choice = st.selectbox("Format", ["PNG", "JPG", "WEBP"], key="download_format")
                with col2:
                    quality_choice = st.selectbox("Quality", ["high", "medium"], key="download_quality")
                
                # Calculate price
                price = payment_handler.calculate_amount(1, quality_choice) / 100
                st.markdown(f"**Price: ₹{price:.2f}**")
                
                if st.button("💳 Purchase & Download", use_container_width=True, type="primary"):
                    # Prepare download for payment
                    download_path, filename = DownloadPreparation.prepare_download(
                        image=styled,
                        user_id=st.session_state.current_user.get('username', 'guest'),
                        original_filename='premium.png',
                        style_name=st.session_state.active_preset,
                        format=format_choice,
                        quality=quality_choice,
                        add_watermark=False
                    )
                    
                    if download_path:
                        st.session_state.download_info = {
                            'path': download_path,
                            'filename': filename,
                            'style': st.session_state.active_preset,
                            'format': format_choice
                        }
                        st.session_state.download_quality = quality_choice
                        st.session_state.payment_page = True
                        st.rerun()

        with canvas_col:
            if "uploaded_image" not in st.session_state or not st.session_state.uploaded_image:
                st.markdown("""
                <div style="
                    height:500px;
                    border:2px dashed #e0e0e0;
                    border-radius:20px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-size:1.2rem;
                    color:#666;
                    background:#f9f9f9;
                ">🎨 Upload a photo to get started</div>
                """, unsafe_allow_html=True)
            else:
                img = st.session_state.uploaded_image
                styled = st.session_state.styled_image or PRESET_FILTERS[st.session_state.active_preset](
                    img, line_w, smooth, detail, color_simp
                )
                w, h = img.size
                
                # Image info bar
                st.markdown(f"""
                <div style="margin-bottom: 1rem; padding:0.5rem; background:#f5f5f5; border-radius:8px;">
                    📐 {w}×{h}px | 🎨 {st.session_state.active_preset}
                </div>
                """, unsafe_allow_html=True)

                if st.session_state.view_mode == "Split":
                    c1, c2 = st.columns(2)
                    with c1:
                        st.markdown("**ORIGINAL**")
                        st.image(img, use_container_width=True)
                    with c2:
                        st.markdown(f"**{st.session_state.active_preset}**")
                        st.image(styled, use_container_width=True)
                else:
                    st.image(styled, use_container_width=True)

if __name__ == "__main__":
    # Run cleanup on startup (optional)
    DownloadPreparation.cleanup_old_files()
    show_editor()