# brain_tumor_webui_pdf.py - Web UI with rich PDF report generation

from flask import Flask, request, render_template_string, url_for, send_file
from ultralytics import YOLO
import os
from PIL import Image
from fpdf import FPDF

app = Flask(__name__)
model = YOLO("best.pt")
UPLOAD_FOLDER = "uploads"
PREDICT_FOLDER = "static/predictions"
PDF_FOLDER = "static/reports"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICT_FOLDER, exist_ok=True)
os.makedirs(PDF_FOLDER, exist_ok=True)

CLASS_NAMES = ['glioma', 'meningioma', 'pituitary']  # adjust based on your model

HTML_TEMPLATE = """
<!doctype html>
<title>Brain Tumor Detection</title>
<h2>Upload an MRI Image</h2>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if image_url %}
  <h3>Prediction Result:</h3>
  <img src="{{ image_url }}" width="500">
  <br><br>
  <a href="{{ pdf_url }}" target="_blank">Download PDF Report</a>
{% endif %}
"""

@app.route('/', methods=['GET', 'POST'])
def index():
    image_url = None
    pdf_url = None

    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename:
            input_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(input_path)

            result = model(input_path)
            filename_base = os.path.splitext(file.filename)[0]
            pred_filename = filename_base + ".jpg"
            output_path = os.path.join(PREDICT_FOLDER, pred_filename)
            result[0].save(filename=output_path)

            # Extract detection info
            pdf_path = os.path.join(PDF_FOLDER, filename_base + ".pdf")
            boxes = result[0].boxes

            # Create PDF (remove emojis for encoding safety)
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "AI Diagnostic Report", ln=True, align='C')
            pdf.set_font("Arial", size=12)
            pdf.ln(5)

            if boxes and boxes.cls.numel() > 0:
                box = boxes[0]  # Show top-1 prediction
                class_id = int(box.cls[0].item())
                label = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"Class {class_id}"
                conf = float(box.conf[0].item())
                coords = box.xyxy[0].tolist()
                width = abs(coords[2] - coords[0])
                height = abs(coords[3] - coords[1])
                size_percent = (width * height) / (640 * 640) * 100  # assume 640x640 input

                pdf.multi_cell(
                    0, 10,
                    f"Tumor Type: {label.title()}\n"
                    f"Tumor Location: Unknown\n"
                    f"Tumor Size: {size_percent:.1f}% of image\n"
                    f"Confidence Score: {conf:.2f}\n"
                    f"Priority Level: {'HIGH' if conf > 0.5 else 'LOW'}"
                )
                pdf.ln(10)
                pdf.set_font("Arial", style="I", size=11)
                pdf.multi_cell(0, 10, "Recommendation:\nUrgent referral to oncology. Advanced imaging required.")
            else:
                pdf.cell(0, 10, "No tumor detected with high confidence.", ln=True)

            pdf.ln(10)
            pdf.image(output_path, x=10, w=180)
            pdf.output(pdf_path, "F")

            image_url = url_for('static', filename=f'predictions/{pred_filename}')
            pdf_url = url_for('static', filename=f'reports/{filename_base}.pdf')

    return render_template_string(HTML_TEMPLATE, image_url=image_url, pdf_url=pdf_url)

if __name__ == '__main__':
    app.run(debug=True, port=5000)