import cv2
import numpy as np
from PIL import Image
import os
import uuid

try:
    import pytesseract
except ImportError:
    pytesseract = None

# ---------------------- GLOBAL --------------------------- #
scanned_pages = []
# ---------------------- UTILS --------------------------- #
def resize_image(img, max_width=800, max_height=600):
    h, w = img.shape[:2]
    scale = min(max_width / w, max_height / h)
    return cv2.resize(img, (int(w * scale), int(h * scale))), scale

def reorder_points(pts):
    pts = np.array(pts, dtype=np.float32).reshape(4, 2)
    new_pts = np.zeros((4, 2), dtype=np.float32)
    add = pts.sum(axis=1)
    new_pts[0] = pts[np.argmin(add)]   # top-left
    new_pts[2] = pts[np.argmax(add)]   # bottom-right
    diff = np.diff(pts, axis=1)
    new_pts[1] = pts[np.argmin(diff)]  # top-right
    new_pts[3] = pts[np.argmax(diff)]  # bottom-left
    return new_pts

def warp_image(img, points, width=250, height=450):
    pts1 = reorder_points(points)
    pts2 = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, matrix, (width, height))

def decode_image_from_bytes(img_bytes):
    npimg = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(npimg, cv2.IMREAD_COLOR)

def encode_image_to_jpg_bytes(img):
    success, buffer = cv2.imencode(".jpg", img)
    if not success:
        return None
    return buffer.tobytes()

def image_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def image_to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ------------------- PAGE STORAGE ---------------------------- #
def add_page(warped):
    scanned_pages.append(warped)
    return len(scanned_pages)

def get_page_count():
    return len(scanned_pages)

def clear_pages():
    scanned_pages.clear()
    return True

def save_all_images(output_folder="saved_pages"):
    if not scanned_pages:
        return None
    os.makedirs(output_folder, exist_ok=True)
    saved_files = []
    for i, page in enumerate(scanned_pages):
        filename = os.path.join(output_folder, f"page_{i + 1}.jpg")
        cv2.imwrite(filename, page, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_files.append(filename)
    return saved_files

def save_as_pdf(filename="final.pdf"):
    if not scanned_pages:
        return None
    pil_images = []
    for img in scanned_pages:
        img_rgb = image_to_rgb(img)
        pil_images.append(Image.fromarray(img_rgb))
    pil_images[0].save(filename, save_all=True, append_images=pil_images[1:])
    return filename

def save_temp_image(img, folder="temp"):
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{uuid.uuid4().hex}.jpg")
    cv2.imwrite(filename, img)
    return filename

# ------------------- DOCUMENT DETECTION ---------------------------- #
def find_document_contour(img):
    gray = image_to_gray(img)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4, 2).tolist()
    return None

def detect_document_points(img):
    img_resized, scale = resize_image(img)
    points = find_document_contour(img_resized)
    return {"success": points is not None,"resized": img_resized,"points": points,"scale": scale}

def draw_detected_border(img, points, color=(59, 130, 246), thickness=3):
    output = img.copy()
    if points is None or len(points) != 4:
        return output
    pts = reorder_points(points).astype(int)
    cv2.polylines(output, [pts], isClosed=True, color=color, thickness=thickness)
    for x, y in pts:
        cv2.circle(output, (int(x), int(y)), 7, (255, 255, 255), -1)
        cv2.circle(output, (int(x), int(y)), 5, color, -1)
    return output

# ------------------- SCAN ---------------------------- #
def auto_detect_only(img):
    result = detect_document_points(img)
    resized = result["resized"]
    points = result["points"]
    overlay = draw_detected_border(resized, points)
    return { "success": points is not None, "image": resized,"points": points,"overlay": overlay}

def scan_with_points(img, points, width=250, height=450):
    if points is None or len(points) != 4:
        return None
    img_resized, _ = resize_image(img)
    warped = warp_image(img_resized, points, width, height)
    return warped

def auto_scan(img, width=250, height=450):
    detected = detect_document_points(img)
    if not detected["success"]:
        return { "success": False,"points": None, "warped": None,"overlay": detected["resized"]}
    warped = warp_image(detected["resized"], detected["points"], width, height)
    overlay = draw_detected_border(detected["resized"], detected["points"])
    return {"success": True, "points": detected["points"],"warped": warped,"overlay": overlay}

# ------------------- FILTERS ---------------------------- #
def apply_filter(img, filter_name="original"):
    if img is None:
        return None
    filter_name = (filter_name or "original").lower()
    if filter_name == "original":
        return img.copy()
    if filter_name == "grayscale":
        gray = image_to_gray(img)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if filter_name == "threshold":
        gray = image_to_gray(img)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    if filter_name == "adaptive":
        gray = image_to_gray(img)
        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
        return cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR)
    if filter_name == "sharpen":
        kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])
        return cv2.filter2D(img, -1, kernel)
    if filter_name == "brighten":
        return cv2.convertScaleAbs(img, alpha=1.15, beta=15)
    return img.copy()

def get_available_filters():
    return ["original","grayscale","threshold","adaptive","sharpen", "brighten"]

# ------------------- OCR ---------------------------- #
def extract_text_from_image(img):
    if img is None:
        return {"success": False,"text": "","message": "No image provided"}
    if pytesseract is None:
        return {"success": False,"text": "","message": "pytesseract is not installed"}
    gray = image_to_gray(img)
    processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 11, 2)
    text = pytesseract.image_to_string(processed)
    return { "success": True,"text": text.strip(),"message": "OCR completed"}