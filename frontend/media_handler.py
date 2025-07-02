import os
from PIL import Image
import streamlit as st
import pandas as pd
import PyPDF2
from datetime import datetime

class MediaHandler:
    def __init__(self):
        self.upload_dir = "frontend/uploads"
        if not os.path.exists(self.upload_dir):
            os.makedirs(self.upload_dir)

    def check_dpi(self, image: Image.Image) -> float:
        try:
            dpi = image.info.get('dpi', (72, 72))[0]
            return dpi
        except:
            return 72

    def enhance_dpi(self, image: Image.Image) -> Image.Image:
        if self.check_dpi(image) < 300:
            width, height = image.size
            new_width = int(width * (300 / self.check_dpi(image)))
            new_height = int(height * (300 / self.check_dpi(image)))
            return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        return image

    def handle_upload(self, uploaded_file, username: str) -> str:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.upload_dir, f"{username}_{timestamp}{file_extension}")

        try:
            if file_extension in ['.jpg', '.png']:
                image = Image.open(uploaded_file)
                if self.check_dpi(image) < 300:
                    image = self.enhance_dpi(image)
                image.save(file_path, dpi=(300, 300))
                return file_path
            elif file_extension == '.xlsx':
                pd.read_excel(uploaded_file)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                return file_path
            else:
                return None
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            return None