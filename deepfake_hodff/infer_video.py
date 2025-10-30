# deepfake_hodff/infer_video.py
import argparse, cv2, torch, numpy as np
from pathlib import Path
from facenet_pytorch import MTCNN
from deepfake_hodff.models import HODFF_DD
from deepfake_hodff.config import *

@torch.no_grad()
def infer_on_video(video_path: Path, out_path: Path):
    mtcnn = MTCNN(select_largest=True, device=DEVICE if DEVICE=="cuda" else "cpu", post_process=False)
    model = HODFF_DD().to(DEVICE)
    model.load_state_dict(torch.load(CKPT_DIR/"hodff_dd.pt", map_location=DEVICE))
    model.eval()

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(round(fps / FPS)))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    frames_face = []
    overlay_frames = []
    idx, saved = 0, 0
    while True:
        ret = cap.grab()
        if not ret: break
        if idx % step == 0:
            ret, frame = cap.retrieve()
            if not ret: break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            boxes, probs = mtcnn.detect(rgb)
            if boxes is not None and len(boxes):
                areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                j = int(np.argmax(areas))
                x1,y1,x2,y2 = boxes[j].astype(int)
                x1,y1,x2,y2 = max(0,x1-FACE_MARGIN), max(0,y1-FACE_MARGIN), min(w,x2+FACE_MARGIN), min(h,y2+FACE_MARGIN)
                face = rgb[y1:y2, x1:x2].copy()
                face224 = cv2.resize(face, (FACE_SIZE_1, FACE_SIZE_1), interpolation=cv2.INTER_AREA)
                face300 = cv2.resize(face224, (FACE_SIZE_2, FACE_SIZE_2), interpolation=cv2.INTER_LINEAR)
                frames_face.append(face300.astype(np.float32)/255.0)
                overlay_frames.append((frame, (x1,y1,x2,y2)))
                saved += 1
        idx += 1
    cap.release()

    if len(frames_face)==0:
        print("No faces detected.")
        return

    # batch into sequences
    X = np.stack(frames_face, axis=0)   # N,H,W,C
    # pad to multiples of SEQ_LEN
    pad = (SEQ_LEN - (len(X)%SEQ_LEN)) % SEQ_LEN
    if pad>0: X = np.concatenate([X, X[-1:].repeat(pad,0)], axis=0)
    X = X.reshape(-1, SEQ_LEN, FACE_SIZE_2, FACE_SIZE_2, 3)
    X = np.transpose(X, (0,1,4,2,3))    # B,T,3,H,W
    x = torch.from_numpy(X).to(DEVICE)

    logits = model(x)                   # (B,T,2)
    probs = torch.softmax(logits, dim=-1)[...,1].reshape(-1).cpu().numpy()[:len(frames_face)]
    maj = int(np.round((probs>=0.5).mean()))
    label = "FAKE" if maj==1 else "REAL"
    conf  = float(np.mean(probs if maj==1 else 1-probs))

    # overlay
    for i, (frame, box) in enumerate(overlay_frames):
        x1,y1,x2,y2 = box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0) if label=="REAL" else (0,0,255), 2)
        cv2.putText(frame, f"{label}  conf={conf:.2f}", (x1, max(20,y1-10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0) if label=="REAL" else (0,0,255), 2)
        out.write(frame)
    out.release()
    print(f"Predicted {label} with conf={conf:.3f}. Saved overlay to {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    infer_on_video(args.video, args.out)

if __name__ == "__main__":
    main()
