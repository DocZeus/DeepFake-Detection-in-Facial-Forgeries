# deepfake_hodff/serve.py
from flask import Flask, request, jsonify
from pathlib import Path
import tempfile
from deepfake_hodff.infer_video import infer_on_video

app = Flask(__name__)

@app.post("/predict")
def predict():
    if "file" not in request.files:
        return jsonify({"error":"no file"}), 400
    f = request.files["file"]
    with tempfile.TemporaryDirectory() as td:
        in_path  = Path(td)/"in.mp4"
        out_path = Path(td)/"out.mp4"
        f.save(in_path)
        infer_on_video(in_path, out_path)
        # you can stream out_path if needed; here we return just metadata
        return jsonify({"status":"ok", "overlay_path": str(out_path)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
