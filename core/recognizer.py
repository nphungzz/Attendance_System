# ============================================================
#  core/recognizer.py  –  ArcFace Embedding + Cosine Similarity
# ============================================================

import pickle
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from config import (
    INSIGHTFACE_MODEL, INSIGHTFACE_CTX_ID,
    EMBEDDINGS_PATH, COSINE_THRESHOLD
)


class FaceRecognizer:
    """
    - Trích xuất 512-D embedding bằng ArcFace (InsightFace buffalo_l).
    - So sánh cosine similarity với database để nhận diện.
    """

    def __init__(self):
        # Khởi tạo InsightFace (chỉ dùng recognition, không dùng detector)
        self.app = FaceAnalysis(name=INSIGHTFACE_MODEL)
        self.app.prepare(ctx_id=INSIGHTFACE_CTX_ID, det_size=(640, 640))

        # Load database embeddings
        self.database: list[dict] = []
        self.load_database()

    # ── Embedding ────────────────────────────────────────────

    def get_embedding(self, face_img) -> np.ndarray | None:
        """
        Trích xuất 512-D embedding từ ảnh khuôn mặt BGR.

        Returns:
            numpy array shape (512,) đã L2-normalize, hoặc None nếu lỗi.
        """
        # InsightFace cần ảnh RGB
        import cv2
        rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        faces = self.app.get(rgb)
        if not faces:
            return None

        emb = faces[0].embedding
        # L2 normalize để dùng dot product thay cho cosine đầy đủ
        norm = np.linalg.norm(emb)
        if norm == 0:
            return None
        return emb / norm

    # ── Database ─────────────────────────────────────────────

    def load_database(self):
        """Load embeddings.pkl vào bộ nhớ."""
        try:
            with open(EMBEDDINGS_PATH, "rb") as f:
                self.database = pickle.load(f)
            print(f"[Recognizer] Loaded {len(self.database)} embeddings from database.")
        except FileNotFoundError:
            self.database = []
            #print("[Recognizer] WARNING: embeddings.pkl not found. Run build_embeddings.py first.")

    def reload_database(self):
        """Reload lại database (dùng khi thêm sinh viên mới)."""
        self.load_database()

    # ── Recognition ──────────────────────────────────────────

    def recognize(self, face_img) -> dict:
        """
        Nhận diện khuôn mặt.

        Returns:
            {
                "name"       : str,     # student ID hoặc "Unknown"
                "similarity" : float,   # cosine similarity cao nhất
                "known"      : bool
            }
        """
        if not self.database:
            return {"name": "Unknown", "similarity": 0.0, "known": False}

        query = self.get_embedding(face_img)
        if query is None:
            return {"name": "Unknown", "similarity": 0.0, "known": False}

        # Tính cosine similarity với tất cả (vì đã normalize: dot = cosine)
        db_embeddings = np.array([entry["embedding"] for entry in self.database])
        similarities  = np.dot(db_embeddings, query)

        best_idx  = int(np.argmax(similarities))
        best_sim  = float(similarities[best_idx])
        best_name = self.database[best_idx]["name"]

        if best_sim >= COSINE_THRESHOLD:
            return {"name": best_name, "similarity": best_sim, "known": True}
        else:
            return {"name": "Unknown", "similarity": best_sim, "known": False}
