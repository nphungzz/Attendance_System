# ============================================================
#  build_embeddings.py  –  Giai đoạn 1: Build Database
#
#  Dùng khi:
#    - Lần đầu thiết lập hệ thống
#    - Thêm sinh viên mới vào dataset
#
#  Chạy:
#    python build_embeddings.py
# ============================================================

import os
import pickle
import cv2
import numpy as np
from insightface.app import FaceAnalysis

from config import (
    DATASET_DIR, EMBEDDINGS_PATH,
    INSIGHTFACE_MODEL, INSIGHTFACE_CTX_ID,
)

# Upscale ảnh nhỏ về ít nhất kích thước này (px cạnh dài)
MIN_SIZE = 300


def build_embeddings():
    print("=" * 55)
    print("  BUILD EMBEDDINGS DATABASE")
    print("=" * 55)

    # ── Kiểm tra dataset ─────────────────────────────────────
    if not os.path.isdir(DATASET_DIR):
        print(f"[ERROR] Dataset directory not found: {DATASET_DIR}")
        return

    image_files = [
        f for f in os.listdir(DATASET_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if not image_files:
        print("[ERROR] No images found in dataset directory.")
        return

    print(f"[Info] Found {len(image_files)} images in dataset.")

    # ── Khởi tạo InsightFace ──────────────────────────────────
    # InsightFace buffalo_l tự detect + align + embed bên trong
    # → không cần crop thủ công, tránh lỗi với ảnh thẻ nhỏ
    app = FaceAnalysis(name=INSIGHTFACE_MODEL)
    app.prepare(ctx_id=INSIGHTFACE_CTX_ID, det_size=(640, 640))

    # ── Xử lý từng ảnh ───────────────────────────────────────
    database = []
    failed   = []

    for fname in sorted(image_files):
        student_id = os.path.splitext(fname)[0]
        img_path   = os.path.join(DATASET_DIR, fname)

        img = cv2.imread(img_path)
        if img is None:
            print(f"  [SKIP] Cannot read: {fname}")
            failed.append(fname)
            continue

        # ── Upscale ảnh thẻ nhỏ ──────────────────────────────
        # Ảnh thẻ SV thường rất nhỏ (< 200px).
        # Scale lên để InsightFace detect mặt chính xác hơn.
        h_img, w_img = img.shape[:2]
        long_side = max(h_img, w_img)
        if long_side < MIN_SIZE:
            scale = MIN_SIZE / long_side
            img = cv2.resize(img, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_CUBIC)
            print(f"  [Info] Upscaled {fname}: "
                  f"{w_img}x{h_img} → {img.shape[1]}x{img.shape[0]}")

        # ── InsightFace: detect + align + embed ──────────────
        # Dùng ảnh gốc (đã upscale), KHÔNG crop thủ công.
        # InsightFace buffalo_l tự lo phần detect & align bên trong.
        rgb   = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = app.get(rgb)

        if not faces:
            print(f"  [SKIP] No face found by InsightFace: {fname}")
            failed.append(fname)
            continue

        # Nếu nhiều mặt → lấy mặt có bbox lớn nhất
        if len(faces) > 1:
            faces = sorted(faces,
                           key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]),
                           reverse=True)

        emb  = faces[0].embedding
        norm = np.linalg.norm(emb)
        if norm == 0:
            print(f"  [SKIP] Zero-norm embedding: {fname}")
            failed.append(fname)
            continue

        emb = emb / norm  # L2 normalize

        database.append({
            "name"     : student_id,
            "embedding": emb.tolist()
        })
        print(f"  [OK]   {student_id}")

    # ── Lưu database ─────────────────────────────────────────
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    with open(EMBEDDINGS_PATH, "wb") as f:
        pickle.dump(database, f)

    print()
    print(f"[Done] Saved {len(database)} embeddings → {EMBEDDINGS_PATH}")
    if failed:
        print(f"[Warn] {len(failed)} images skipped: {failed}")
    print("=" * 55)


if __name__ == "__main__":
    build_embeddings()
