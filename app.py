from flask import Flask, render_template, request, jsonify, send_file
import base64
import os

from scanner import (
    decode_image_from_bytes,
    encode_image_to_jpg_bytes,
    auto_detect_only,
    scan_with_points,
    apply_filter,
    get_available_filters,
    add_page,
    get_page_count,
    save_as_pdf,
    save_all_images,
    extract_text_from_image,
)

app = Flask(__name__)

current_image = None
current_points = None
current_scanned = None
current_filtered = None


def image_to_base64(img):
    img_bytes = encode_image_to_jpg_bytes(img)
    if img_bytes is None:
        return None
    return base64.b64encode(img_bytes).decode("utf-8")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/filters", methods=["GET"])
def filters():
    return jsonify({
        "success": True,
        "filters": get_available_filters()
    })


@app.route("/upload", methods=["POST"])
def upload_image():
    global current_image, current_points, current_scanned, current_filtered

    file = request.files.get("image")
    if not file:
        return jsonify({
            "success": False,
            "message": "No image received"
        })

    img = decode_image_from_bytes(file.read())
    if img is None:
        return jsonify({
            "success": False,
            "message": "Invalid image file"
        })

    current_image = img
    current_points = None
    current_scanned = None
    current_filtered = None

    return jsonify({
        "success": True,
        "message": "Image uploaded successfully"
    })


@app.route("/auto-detect", methods=["POST"])
def auto_detect():
    global current_image, current_points, current_scanned, current_filtered

    if current_image is None:
        return jsonify({
            "success": False,
            "message": "No image available"
        })

    result = auto_detect_only(current_image)

    if not result["success"]:
        current_points = None
        current_scanned = None
        current_filtered = None
        return jsonify({
            "success": False,
            "message": "Auto detection failed"
        })

    current_points = result["points"]
    current_scanned = None
    current_filtered = None

    return jsonify({
        "success": True,
        "message": "Border detected. Review it, then scan or switch to manual adjust.",
        "points": result["points"],
        "image": image_to_base64(result["overlay"])
    })


@app.route("/scan", methods=["POST"])
def scan_document():
    global current_image, current_points, current_scanned, current_filtered

    if current_image is None:
        return jsonify({
            "success": False,
            "message": "No image available"
        })

    data = request.get_json(silent=True) or {}
    points = data.get("points", current_points)

    if points is None or len(points) != 4:
        return jsonify({
            "success": False,
            "message": "No valid points available for scanning"
        })

    scanned = scan_with_points(current_image, points)

    if scanned is None:
        return jsonify({
            "success": False,
            "message": "Scanning failed"
        })

    current_points = points
    current_scanned = scanned
    current_filtered = scanned.copy()

    return jsonify({
        "success": True,
        "message": "Document scanned successfully",
        "image": image_to_base64(current_filtered)
    })


@app.route("/manual-adjust", methods=["POST"])
def manual_adjust():
    global current_points

    data = request.get_json(silent=True) or {}
    points = data.get("points")

    if points is None or len(points) != 4:
        return jsonify({
            "success": False,
            "message": "Please provide 4 corner points"
        })

    current_points = points

    return jsonify({
        "success": True,
        "message": "Manual points updated",
        "points": current_points
    })


@app.route("/apply-filter", methods=["POST"])
def apply_selected_filter():
    global current_scanned, current_filtered

    if current_scanned is None:
        return jsonify({
            "success": False,
            "message": "No scanned image available"
        })

    data = request.get_json(silent=True) or {}
    filter_name = data.get("filter", "original")

    filtered = apply_filter(current_scanned, filter_name)
    if filtered is None:
        return jsonify({
            "success": False,
            "message": "Filter could not be applied"
        })

    current_filtered = filtered

    return jsonify({
        "success": True,
        "message": f"Filter applied: {filter_name}",
        "image": image_to_base64(current_filtered)
    })


@app.route("/add-page", methods=["POST"])
def add_scanned_page():
    global current_filtered, current_scanned

    image_to_store = current_filtered if current_filtered is not None else current_scanned

    if image_to_store is None:
        return jsonify({
            "success": False,
            "message": "No scanned page to add"
        })

    total_pages = add_page(image_to_store)

    return jsonify({
        "success": True,
        "message": "Page added successfully",
        "total_pages": total_pages
    })


@app.route("/page-count", methods=["GET"])
def page_count():
    return jsonify({
        "success": True,
        "total_pages": get_page_count()
    })


@app.route("/save-pdf", methods=["GET"])
def save_pdf():
    pdf_path = save_as_pdf("final.pdf")

    if pdf_path is None or not os.path.exists(pdf_path):
        return jsonify({
            "success": False,
            "message": "No pages to save"
        })

    return send_file(pdf_path, as_attachment=True)


@app.route("/save-images", methods=["GET"])
def save_images():
    saved_files = save_all_images()

    if saved_files is None:
        return jsonify({
            "success": False,
            "message": "No pages to save"
        })

    return jsonify({
        "success": True,
        "message": "Images saved successfully",
        "files": saved_files
    })


@app.route("/ocr", methods=["POST"])
def run_ocr():
    global current_filtered, current_scanned

    target = current_filtered if current_filtered is not None else current_scanned

    if target is None:
        return jsonify({
            "success": False,
            "message": "No scanned image available for OCR",
            "text": ""
        })

    result = extract_text_from_image(target)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)