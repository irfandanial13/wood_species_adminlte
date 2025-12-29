from flask import (
    Flask, render_template, request, session,
    send_from_directory, jsonify, abort
)
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import tensorflow as tf
import re
from difflib import get_close_matches
from datetime import datetime
from werkzeug.utils import secure_filename
import uuid

# ======================================================
# 🌳 FLASK CONFIG
# ======================================================
app = Flask(__name__)
app.secret_key = os.getenv("APP_SECRET_KEY", "danial_fyp_secret")

UPLOAD_FOLDER = 'static/uploads'
DATA_RAW_DIR = 'data/raw'
TREE_IMAGE_DIR = 'data/tree_images'
INFO_CSV = 'data/wood_species_info.csv'
MODEL_PATH = 'models/tree_species_model.h5'

# CSV prediction history
PREDICTION_CSV = r"C:\wood_species_adminlte\data\prediction_history.csv"

# ensure folders exist
for folder in [UPLOAD_FOLDER, DATA_RAW_DIR, TREE_IMAGE_DIR]:
    os.makedirs(folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ======================================================
# 🧠 LOAD MODEL & CSV
# ======================================================
def load_model_safe():
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None

model = load_model_safe()

def clean_text(val):
    if isinstance(val, str):
        val = val.replace('\xa0', ' ').replace('\x96', '-')
        val = re.sub(r'[^\x00-\x7F]+', ' ', val).strip()
    return val

try:
    info_df = pd.read_csv(INFO_CSV, encoding='latin1', engine='python')
    info_df.columns = [c.strip() for c in info_df.columns]
    for col in info_df.select_dtypes(include=['object']).columns:
        info_df[col] = info_df[col].astype(str).map(clean_text)
except:
    info_df = pd.DataFrame()

CLASS_NAMES = sorted([p.name for p in os.scandir(DATA_RAW_DIR) if p.is_dir()])

# ======================================================
# UTILS
# ======================================================
def to_filename_base(text):
    if not text:
        return ""
    s = str(text).lower()
    s = re.sub(r'[^a-z0-9]+', '_', s).strip('_')
    return s or "unknown"

def save_prediction_to_csv(record: dict):
    os.makedirs(os.path.dirname(PREDICTION_CSV), exist_ok=True)
    df = pd.DataFrame([record])
    if not os.path.exists(PREDICTION_CSV):
        df.to_csv(PREDICTION_CSV, index=False)
    else:
        df.to_csv(PREDICTION_CSV, mode='a', header=False, index=False)

# ======================================================
# IMAGE PREPROCESS
# ======================================================
def preprocess_image(img):
    img = ImageOps.fit(img, (224, 224), Image.BICUBIC)
    arr = np.asarray(img).astype("float32") / 255.0
    return np.expand_dims(arr, 0)

# ======================================================
# STATIC
# ======================================================
@app.route('/tree_images/<path:filename>')
def tree_images(filename):
    filename = filename.replace("%20", " ")
    try:
        return send_from_directory(TREE_IMAGE_DIR, filename)
    except:
        return send_from_directory("static", "default_tree.jpg")

@app.route('/data/raw/<path:filename>')
def data_raw(filename):
    try:
        return send_from_directory(DATA_RAW_DIR, filename)
    except:
        abort(404)

# ======================================================
# SPECIES MATCH
# ======================================================
def get_sci_col():
    matches = [c for c in info_df.columns if "scientific" in c.lower()]
    return matches[0] if matches else info_df.columns[0] if len(info_df.columns) else None

def match_species_info(label):
    if info_df.empty:
        return None

    clean = str(label).lower().strip()
    sci_col = get_sci_col()
    if not sci_col:
        return None

    df = info_df.copy()
    df[sci_col] = df[sci_col].astype(str)

    exact = df[df[sci_col].str.lower().str.strip() == clean]
    if not exact.empty:
        return exact.iloc[0].to_dict()

    partial = df[df[sci_col].str.lower().str.contains(re.escape(clean), na=False)]
    if not partial.empty:
        return partial.iloc[0].to_dict()

    names = df[sci_col].str.lower().str.strip().tolist()
    closest = get_close_matches(clean, names, n=1, cutoff=0.5)
    if closest:
        row = df[df[sci_col].str.lower().str.strip() == closest[0]]
        if not row.empty:
            return row.iloc[0].to_dict()

    return None

# ======================================================
# 🔮 PREDICTION ENGINE
# ======================================================
def process_prediction(image_path, is_uploaded=False):
    if model is None:
        return {"error": "Model not loaded"}

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"error": f"Failed to open image: {e}"}

    arr = preprocess_image(img)
    preds = model.predict(arr, verbose=0)[0]

    top_idx = preds.argsort()[-3:][::-1]
    raw_labels = [CLASS_NAMES[i] for i in top_idx]
    labels = [lbl.split("-", 1)[1].strip() if "-" in lbl else lbl for lbl in raw_labels]
    confs = [round(float(preds[i] * 100), 2) for i in top_idx]

    top1 = labels[0]
    matched = match_species_info(top1)

    search_name = matched.get("Scientific Name", top1) if matched else top1
    base = to_filename_base(search_name)

    species_image = "/static/default_tree.jpg"
    for ext in [".jpg", ".jpeg", ".png"]:
        path = os.path.join(TREE_IMAGE_DIR, f"{base}{ext}")
        if os.path.exists(path):
            species_image = f"/tree_images/{base}{ext}"
            break

    image_url = "/" + image_path.replace("\\", "/")

    history = session.get("history", [])
    record_id = str(uuid.uuid4())
    history.insert(0, {
        "id": record_id,
        "image_path": image_url,
        "predicted_species": top1,
        "confidence": int(confs[0]),
        "created_at": datetime.now().strftime("%d %b %Y, %I:%M %p")
    })
    session["history"] = history

    save_prediction_to_csv({
        "id": record_id,
        "image_path": image_url,
        "predicted_species": top1,
        "confidence_percent": int(confs[0]),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

    return {
        "filename": os.path.basename(image_path),
        "label": top1,
        "confidence": int(confs[0]),
        "matched": matched or {},
        "top_labels": labels,
        "top_conf": confs,
        "species_image": species_image,
    }

# ======================================================
# ROUTES
# ======================================================
@app.route('/')
def index():
    samples = []
    for folder in CLASS_NAMES:
        folder_path = os.path.join(DATA_RAW_DIR, folder)
        if not os.path.isdir(folder_path):
            continue
        imgs = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        if imgs:
            clean = folder.split("-",1)[1] if "-" in folder else folder
            samples.append({
                "img": f"/data/raw/{folder}/{imgs[0]}",
                "path": f"{folder}/{imgs[0]}",
                "name": clean
            })
    return render_template("index.html", sample_images=samples)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get("file")
    if not file:
        return render_template("index.html", error="No file selected")

    fname = secure_filename(file.filename) or f"upload_{uuid.uuid4().hex}.jpg"
    final_name = f"{uuid.uuid4().hex}_{fname}"
    fpath = os.path.join(UPLOAD_FOLDER, final_name)
    file.save(fpath)

    result = process_prediction(fpath, is_uploaded=True)
    return render_template("result.html", **result)

@app.route('/predict_sample/<path:image_path>')
def predict_sample(image_path):
    safe = os.path.normpath(image_path)
    if safe.startswith("..") or os.path.isabs(safe):
        return "Invalid path", 400

    full = os.path.join(DATA_RAW_DIR, safe)
    if not os.path.exists(full):
        return "Sample not found", 404

    result = process_prediction(full)
    return render_template("result.html", **result)

@app.route('/dashboard')
def dashboard():
    history = session.get("history", [])

    # ===============================
    # 🌳 TREE HEIGHT DATA
    # ===============================
    trees = []
    if not info_df.empty and "Tree Height (m)" in info_df.columns:
        for _, row in info_df.iterrows():
            height_str = str(row.get("Tree Height (m)", "")).lower()
            height_str = height_str.replace("Â–", "–").replace("-", "–")

            numbers = re.findall(r'\d+', height_str)
            if not numbers:
                continue

            height = max(map(int, numbers))

            trees.append({
                "name": row.get("Scientific Name", "Unknown"),
                "height": height,
                "svg_height": height * 3
            })

    # ===============================
    # 🌲 WOOD USE CONCENTRATION DATA
    # ===============================
    wood_use_categories = {
        "Construction": 0,
        "Furniture": 0,
        "Fuel / Firewood": 0,
        "Medicinal": 0,
        "Edible": 0,
        "Agroforestry": 0
    }

    if not info_df.empty:
        for _, row in info_df.iterrows():
            text = " ".join([
                str(row.get("Wood Uses", "")),
                str(row.get("Wood Uses (Detailed)", "")),
                str(row.get("Medicinal Uses", "")),
                str(row.get("Edible Uses", "")),
                str(row.get("Agroforestry Uses", ""))
            ]).lower()

            if "construction" in text:
                wood_use_categories["Construction"] += 1
            if "furniture" in text:
                wood_use_categories["Furniture"] += 1
            if "fuel" in text or "firewood" in text:
                wood_use_categories["Fuel / Firewood"] += 1
            if "medicinal" in text and "none" not in text:
                wood_use_categories["Medicinal"] += 1
            if "edible" in text and "none" not in text:
                wood_use_categories["Edible"] += 1
            if "agroforestry" in text:
                wood_use_categories["Agroforestry"] += 1
     # ===============================
    # 🌿 SPECIES RICHNESS BY FAMILY
    # ===============================
    family_counts = {}

    if not info_df.empty and "Family" in info_df.columns:
        for fam in info_df["Family"]:
            fam = str(fam).strip()
            if not fam or fam.lower() in ["unknown", "not documented", "nan"]:
                continue
            family_counts[fam] = family_counts.get(fam, 0) + 1
            
    return render_template(
    "dashboard.html",
    history=history,
    trees=trees,
    wood_use=wood_use_categories,
    family_counts=family_counts
)


@app.route('/clear_history', methods=['POST'])
def clear_history():
    session["history"] = []
    return ('', 204)

@app.route('/delete_history_item', methods=['POST'])
def delete_history_item():
    data = request.get_json()
    id_ = data.get("id")
    history = session.get("history", [])
    session["history"] = [h for h in history if h["id"] != id_]
    return jsonify({"status": "ok"})

@app.route('/species')
def species():
    if info_df.empty:
        return render_template("species_list.html", species=[])

    sci = [c for c in info_df.columns if "scientific" in c.lower()]
    com = [c for c in info_df.columns if "common" in c.lower()]

    sci_col = sci[0] if sci else info_df.columns[0]
    com_col = com[0] if com else None

    data = []
    for _, row in info_df.iterrows():
        data.append({
            "scientific_name": row.get(sci_col, "").strip(),
            "common_name": row.get(com_col, "").strip() if com_col else ""
        })

    return render_template("species_list.html", species=data)

@app.route('/api/species_info/<path:name>')
def api_species_info(name):
    if info_df.empty:
        return jsonify({"error":"No data"}),500

    sci_col = get_sci_col()
    clean = name.replace("%20"," ").strip().lower()

    row = info_df[info_df[sci_col].str.lower().str.strip() == clean]
    if row.empty:
        return jsonify({"error":"Species not found"}),404

    data = row.iloc[0].to_dict()

    sci_name = data.get("Scientific Name", clean)
    base = to_filename_base(sci_name)
    img = "/static/default_tree.jpg"
    for ext in [".jpg",".jpeg",".png"]:
        if os.path.exists(os.path.join(TREE_IMAGE_DIR, f"{base}{ext}")):
            img = f"/tree_images/{base}{ext}"
            break

    data["image"] = img
    return jsonify(data)

# ======================================================
# RUN
# ======================================================
if __name__ == '__main__':
    app.run(debug=True)
