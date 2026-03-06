# components/download_prep.py
import os
import hashlib
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
import io
import logging
import shutil
from typing import Optional, Tuple, Dict, Any
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
BASE_DIR = Path(__file__).parent.parent
OUTPUT_FOLDER = BASE_DIR / "data" / "processed_images"
TEMP_FOLDER = BASE_DIR / "data" / "temp_downloads"
WATERMARK_TEXT = "TOONIFY PREVIEW"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(TEMP_FOLDER, exist_ok=True)

class DownloadPreparation:
    """Handles image processing and download preparation"""
    
    @staticmethod
    def generate_unique_filename(user_id: str, original_filename: str, style_name: str) -> str:
        """Generate a unique filename using user ID, timestamp, and original filename"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Clean filename and get extension
        base_name = os.path.splitext(original_filename)[0]
        ext = os.path.splitext(original_filename)[1] or '.png'
        
        # Create unique hash
        unique_hash = hashlib.md5(f"{user_id}{timestamp}{base_name}".encode()).hexdigest()[:8]
        
        # Sanitize style name
        style_clean = style_name.replace(" ", "_").replace("🎨", "").replace("⚡", "").replace("🖋️", "").strip()
        
        return f"{user_id}_{style_clean}_{timestamp}_{unique_hash}{ext}"
    
    @staticmethod
    def add_watermark(image: Image.Image, text: str = WATERMARK_TEXT, opacity: float = 0.3) -> Image.Image:
        """Add a subtle watermark to the image"""
        # Create a copy of the image
        watermarked = image.copy()
        
        # Create a transparent overlay
        watermark = Image.new('RGBA', watermarked.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(watermark)
        
        # Try to load a font, fallback to default
        try:
            # Calculate font size based on image dimensions
            font_size = int(min(watermarked.size) / 20)
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
            except:
                font = ImageFont.load_default()
        except:
            font = ImageFont.load_default()
        
        # Get text bounding box
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Calculate position (center of image)
        x = (watermarked.width - text_width) // 2
        y = (watermarked.height - text_height) // 2
        
        # Draw text with opacity
        draw.text((x, y), text, fill=(255, 255, 255, int(255 * opacity)), font=font)
        
        # Composite the watermark
        watermarked = Image.alpha_composite(watermarked.convert('RGBA'), watermark)
        
        return watermarked.convert('RGB')
    
    @staticmethod
    def save_image(image: Image.Image, filepath: Path, quality: int = 95, format: str = "PNG") -> bool:
        """Save image with error handling"""
        try:
            if format.upper() == "JPG" or format.upper() == "JPEG":
                image.save(filepath, "JPEG", quality=quality, optimize=True)
            elif format.upper() == "PNG":
                image.save(filepath, "PNG", optimize=True)
            elif format.upper() == "WEBP":
                image.save(filepath, "WEBP", quality=quality, method=6)
            else:
                image.save(filepath, format.upper())
            logger.info(f"Image saved successfully: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return False
    
    @staticmethod
    def prepare_download(image: Image.Image, user_id: str, original_filename: str, 
                        style_name: str, format: str = "PNG", quality: str = "high", 
                        add_watermark: bool = False) -> Tuple[Optional[str], Optional[str]]:
        """
        Prepare image for download
        Returns: (filepath, filename) or (None, None) on failure
        """
        try:
            # Generate unique filename
            ext = ".png" if format.upper() == "PNG" else ".jpg" if format.upper() in ["JPG", "JPEG"] else ".webp"
            filename = DownloadPreparation.generate_unique_filename(user_id, original_filename, style_name)
            filename = os.path.splitext(filename)[0] + ext
            
            # Set quality based on selection
            quality_value = 95 if quality == "high" else 75
            
            # Apply watermark if needed
            if add_watermark:
                image = DownloadPreparation.add_watermark(image)
            
            # Save to temp folder first
            temp_path = TEMP_FOLDER / filename
            if DownloadPreparation.save_image(image, temp_path, quality_value, format):
                # Store in session state for tracking
                st.session_state['download_info'] = {
                    'filename': filename,
                    'path': str(temp_path),
                    'timestamp': datetime.now().isoformat(),
                    'user_id': user_id,
                    'style': style_name,
                    'format': format,
                    'watermarked': add_watermark
                }
                return str(temp_path), filename
            
            return None, None
            
        except Exception as e:
            logger.error(f"Error preparing download: {e}")
            return None, None
    
    @staticmethod
    def get_download_bytes(filepath: str) -> Optional[bytes]:
        """Get file bytes for download"""
        try:
            with open(filepath, 'rb') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file: {e}")
            return None
    
    @staticmethod
    def cleanup_old_files(hours: int = 24):
        """Clean up files older than specified hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Clean temp folder
            for file_path in TEMP_FOLDER.glob('*'):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        file_path.unlink()
                        logger.info(f"Deleted old temp file: {file_path}")
            
            # Clean output folder
            for file_path in OUTPUT_FOLDER.glob('*'):
                if file_path.is_file():
                    file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if file_time < cutoff_time:
                        file_path.unlink()
                        logger.info(f"Deleted old output file: {file_path}")
                        
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    @staticmethod
    def create_pdf(images: list, output_path: str) -> bool:
        """Create PDF from multiple images"""
        try:
            if len(images) == 1:
                images[0].save(output_path, "PDF", resolution=100.0, save_all=True)
            else:
                images[0].save(output_path, "PDF", resolution=100.0, save_all=True, append_images=images[1:])
            return True
        except Exception as e:
            logger.error(f"Error creating PDF: {e}")
            return False

# Initialize cleanup on module load (optional)
DownloadPreparation.cleanup_old_files()