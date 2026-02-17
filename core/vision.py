import base64
import io
from PIL import ImageGrab


class VisionSystem:
    """Handles screenshot capture for visual context"""

    def capture_screenshot(self):
        """Take a screenshot and convert to base64"""
        screenshot = ImageGrab.grab()
        buffered = io.BytesIO()
        screenshot.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        return img_base64
