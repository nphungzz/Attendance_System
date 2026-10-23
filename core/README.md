# Attendance System – Installation and Usage Guide

## Directory Structure

```text
AttendanceSystem/
├── main.py                  # Run real-time attendance
├── build_embeddings.py      # Build the face embedding database
├── config.py                # Centralized configuration
├── requirements.txt
│
├── core/
│   ├── detector.py          # Face Detection (OpenCV SSD)
│   ├── stability.py         # Stability Check
│   ├── quality.py           # Blur Check
│   ├── recognizer.py        # ArcFace + Cosine Similarity
│   ├── attendance.py        # Save attendance records to CSV
│   └── pipeline.py          # Integrate the entire pipeline
│
├── Data/
│   ├── Dataset/             # Place student images here
│   │   ├── HE191911.jpg
│   │   ├── HE200143.jpg
│   │   └── ...
│   ├── embeddings.pkl       # Generated during database building
│   └── attendance.csv       # Generated during execution
│
└── Models/
    ├── deploy.prototxt
    └── res10_300x300_ssd_iter_140000.caffemodel
```

---

## Installation

Install the required Python packages:

```bash
pip install -r requirements.txt
```

InsightFace will automatically download the `buffalo_l` model to `~/.insightface/models/` the first time it runs.

### Download the OpenCV SSD Models

Run the following commands from the project root directory.

```bash
# Download deploy.prototxt
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt \
    -O Models/deploy.prototxt

# Download the Caffe model
wget https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel \
    -O Models/res10_300x300_ssd_iter_140000.caffemodel
```

---

## How to Run

### Step 1 – Add Student Images

Place student images in `Data/Dataset/`.

Name each image using the corresponding student ID.

Example:

```text
Data/Dataset/HE191911.jpg
Data/Dataset/HE200143.jpg
```

### Step 2 – Build the Face Embedding Database

Run the following command:

```bash
python build_embeddings.py
```

This will generate `Data/embeddings.pkl`, which stores the face embeddings used for recognition.

### Step 3 – Start the Attendance System

Run the following command:

```bash
python main.py
```

### Keyboard Shortcuts

| Key | Function |
|---|---|
| `Q` | Quit the application |
| `R` | Reload the database after adding new students |
| `C` | Print the list of students who have been marked present to the terminal |

---

## Configuration (`config.py`)

| Parameter | Default Value | Description |
|---|---|---|
| `CAMERA_SOURCE` | `0` | Webcam source, or an RTSP stream URL |
| `COSINE_THRESHOLD` | `0.45` | Face recognition threshold; increasing it makes matching stricter |
| `BLUR_THRESHOLD` | `120.0` | Image sharpness threshold; increasing it requires sharper images |
| `STABILITY_MIN_COUNT` | `5` | Number of stable frames required before recognition |
| `INSIGHTFACE_CTX_ID` | `0` | `0` = GPU, `-1` = CPU |

---

## Attendance Records

Attendance records are saved to `Data/attendance.csv`.

Example:

```csv
StudentID,Time,Status
HE191911,2024-01-15 08:30:12,Present
HE200143,2024-01-15 08:31:05,Present
```

Each record contains the student's ID, attendance timestamp, and attendance status.