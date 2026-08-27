````markdown
# Smart Attendance System

> An AI-powered, real-time biometric attendance management system using deep-learning-based face detection and face recognition.

[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Framework-Flask-black.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV-red.svg)](https://opencv.org/)
[![InsightFace](https://img.shields.io/badge/AI-InsightFace-orange.svg)](https://insightface.ai/)
[![SQLite](https://img.shields.io/badge/Database-SQLite-blue.svg)](https://www.sqlite.org/)
[![GitHub](https://img.shields.io/badge/Version%20Control-GitHub-black.svg)](https://github.com/)

---

## 📌 Abstract

The **Smart Attendance System** is an AI-driven biometric attendance management platform designed to automate the process of identifying individuals and recording attendance in real time.

Traditional attendance systems such as manual registers, ID cards, and conventional identification methods can be time-consuming, difficult to manage at scale, and susceptible to proxy attendance. This system addresses these limitations by combining **computer vision, deep-learning-based face detection, face recognition, and database-driven attendance management** into a unified web-based platform.

The system captures facial information through a camera, detects multiple faces in a frame, generates discriminative facial embeddings, compares them against enrolled identities, and automatically records attendance with the corresponding date and time.

The platform is implemented using **Python, Flask, OpenCV, InsightFace, and SQLite**, providing a lightweight architecture suitable for academic demonstrations, institutional environments, and further research into AI-based biometric identification systems.

---

## 🎯 Objectives

The primary objectives of the project are:

- Automate attendance recording using facial biometrics.
- Eliminate the need for manual attendance marking.
- Reduce the possibility of proxy attendance.
- Detect and recognize multiple faces simultaneously.
- Provide real-time attendance processing.
- Maintain persistent attendance records using a database.
- Provide a web-based interface for monitoring and managing attendance.
- Design a modular architecture that can be extended with advanced AI models.
- Evaluate the practical application of deep-learning-based face recognition in real-world attendance scenarios.

---

## 🚀 Key Features

### 👤 Real-Time Face Recognition

The system processes live camera input and identifies enrolled individuals using facial embeddings.

### 👥 Multi-Face Detection

Multiple individuals can be detected and processed within a single camera frame.

### 🧠 Deep Learning-Based Recognition

The recognition pipeline uses modern deep-learning-based face detection and recognition architectures provided through InsightFace.

### ⏱️ Automated Attendance Recording

Once a registered individual is successfully recognized, the system records attendance automatically along with the relevant timestamp.

### 🌐 Web-Based Dashboard

A Flask-based web interface provides an accessible interface for:

- Face enrollment
- Real-time recognition
- Attendance monitoring
- Attendance history
- Database-backed record management

### 🗄️ Persistent Database

Attendance information is stored using SQLite, allowing records to persist across application sessions.

### 📊 Attendance History

The system provides a historical view of attendance records for monitoring and analysis.

### ⚡ Lightweight Architecture

The system is designed to run on conventional computing hardware without requiring dedicated cloud infrastructure.

---

# 🧠 AI & Computer Vision Architecture

The core biometric pipeline consists of two major stages: face detection and face recognition.

```text
             Camera / Image Input
                     │
                     ▼
          ┌─────────────────────┐
          │   Face Detection    │
          │       SCRFD         │
          └──────────┬──────────┘
                     │
                     ▼
          Detected Face Regions
                     │
                     ▼
          ┌─────────────────────┐
          │ Facial Embedding    │
          │    Generation       │
          │    MobileFaceNet    │
          └──────────┬──────────┘
                     │
                     ▼
             Face Embedding
                     │
                     ▼
          ┌─────────────────────┐
          │ Identity Matching   │
          │ Against Enrolled    │
          │    Embeddings       │
          └──────────┬──────────┘
                     │
                     ▼
             Recognized Person
                     │
                     ▼
          ┌─────────────────────┐
          │ Attendance Engine   │
          └──────────┬──────────┘
                     │
                     ▼
              SQLite Database
                     │
                     ▼
             Web Dashboard
````

---

# 🔬 Model Components

## 1. SCRFD — Face Detection

**SCRFD (Sample and Computation Redistribution for Efficient Face Detection)** is used for detecting faces in input images or video frames.

Its role is to:

* Locate faces in an image.
* Generate bounding boxes.
* Support detection of multiple faces.
* Provide efficient face detection suitable for real-time applications.

The detector acts as the first stage of the biometric pipeline.

---

## 2. MobileFaceNet — Face Recognition

After detecting a face, the system extracts a compact facial representation using **MobileFaceNet**.

The model converts a detected face into a numerical **face embedding**.

Conceptually:

```text
Face Image
    ↓
Preprocessing
    ↓
MobileFaceNet
    ↓
Feature Vector / Embedding
    ↓
Similarity Comparison
    ↓
Identity
```

Instead of directly comparing raw images, the system compares the generated feature representations.

---

## 3. InsightFace Model Pack — buffalo_s

The project utilizes the **InsightFace `buffalo_s` model pack**, which provides the required components for face analysis.

The model pack integrates detection and recognition capabilities into a practical inference pipeline.

---

# 🔄 System Workflow

The complete system operates through the following workflow.

## Step 1 — Face Enrollment

A person's face is captured and processed.

```text
Input Face
    ↓
Face Detection
    ↓
Face Alignment / Preprocessing
    ↓
Feature Extraction
    ↓
Face Embedding
    ↓
Store Enrollment Data
```

---

## Step 2 — Live Recognition

During attendance marking, the system receives a live camera frame.

```text
Camera Frame
      ↓
Face Detection
      ↓
Face Cropping
      ↓
Feature Extraction
      ↓
Embedding Generation
      ↓
Similarity Matching
      ↓
Identity Prediction
```

---

## Step 3 — Attendance Verification

If the detected identity matches an enrolled person with sufficient confidence:

```text
Recognized Identity
        ↓
Check Existing Attendance
        ↓
Already Marked?
    ↙           ↘
  YES            NO
   ↓              ↓
Ignore       Record Attendance
                  ↓
             Date + Time
                  ↓
             SQLite DB
```

This prevents repeated attendance entries for the same person within the defined attendance period.

---

# 🏗️ System Architecture

The application follows a modular client-server architecture.

```text
┌──────────────────────────────────────┐
│            User / Camera             │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│             Frontend                 │
│          HTML / CSS / JavaScript     │
└──────────────────┬───────────────────┘
                   │
                   │ HTTP Requests
                   ▼
┌──────────────────────────────────────┐
│             Flask Backend            │
│                                      │
│  ┌────────────────────────────────┐  │
│  │ Face Detection                 │  │
│  │ SCRFD                          │  │
│  └───────────────┬────────────────┘  │
│                  ▼                   │
│  ┌────────────────────────────────┐  │
│  │ Face Recognition               │  │
│  │ MobileFaceNet / InsightFace    │  │
│  └───────────────┬────────────────┘  │
│                  ▼                   │
│  ┌────────────────────────────────┐  │
│  │ Attendance Processing          │  │
│  └───────────────┬────────────────┘  │
└──────────────────┼───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│             SQLite Database          │
│                                      │
│  • User Information                  │
│  • Face Enrollment Data              │
│  • Attendance Records                │
│  • Date / Time Information           │
└──────────────────────────────────────┘
```

---

# 🛠️ Technology Stack

| Category                | Technology            |
| ----------------------- | --------------------- |
| Programming Language    | Python                |
| Web Framework           | Flask                 |
| Computer Vision         | OpenCV                |
| Face Detection          | SCRFD                 |
| Face Recognition        | MobileFaceNet         |
| Face Analysis Framework | InsightFace           |
| Database                | SQLite                |
| Frontend                | HTML, CSS, JavaScript |
| Development Environment | Visual Studio Code    |
| Version Control         | Git & GitHub          |

---

# 📁 Project Structure

```text
smart-attendance-system/
│
├── backend/
│   │
│   ├── main.py
│   │       └── Flask application and API logic
│   │
│   ├── data.py
│   │       └── Database initialization and data operations
│   │
│   ├── templates/
│   │   ├── index.html
│   │   └── history.html
│   │
│   └── database/
│       └── Local SQLite database
│
├── .gitignore
├── README.md
└── requirements.txt
```

> **Note:** Local database files and biometric enrollment data are intentionally excluded from version control through `.gitignore`.

---

# ⚙️ Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/24A31A4209/smart-attendance-system.git
```

```bash
cd smart-attendance-system
```

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
```

Activate it:

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If a requirements file is not available, the major dependencies include:

```bash
pip install flask opencv-python insightface numpy onnxruntime
```

---

## 4. Run the Application

From the project directory:

```bash
python backend/main.py
```

The Flask server will start locally.

Open the displayed local URL in your browser, typically:

```text
http://127.0.0.1:5000
```

---

# 🧪 Testing & Evaluation

The system can be evaluated using:

* Registered individuals
* Unknown individuals
* Multiple faces in a single frame
* Different illumination conditions
* Different face orientations
* Different camera distances
* Repeated attendance attempts

## Suggested Evaluation Metrics

| Metric                | Purpose                                               |
| --------------------- | ----------------------------------------------------- |
| Recognition Accuracy  | Measures correct identity predictions                 |
| Precision             | Measures reliability of positive identity predictions |
| Recall                | Measures ability to correctly identify enrolled users |
| F1-Score              | Harmonic mean of precision and recall                 |
| False Acceptance Rate | Measures incorrect identity acceptance                |
| False Rejection Rate  | Measures failure to recognize registered users        |
| Inference Time        | Measures real-time performance                        |
| FPS                   | Measures video processing efficiency                  |

---

# 📈 Research Perspective

The project demonstrates how modern deep-learning-based biometric systems can be integrated into practical attendance management applications.

The combination of efficient face detection and lightweight face recognition enables the system to perform identity verification without requiring traditional identification mechanisms such as ID cards or manual registers.

From a research perspective, the system provides a foundation for investigating:

* Real-time biometric identification
* Lightweight face recognition
* Edge-based AI inference
* Face recognition under varying illumination
* Multi-face recognition
* Recognition robustness
* Biometric security
* AI-based institutional automation

---

# 🔐 Privacy & Security Considerations

Because the system processes biometric information, privacy and security are important considerations.

The project follows these principles:

* Biometric enrollment data should be stored securely.
* Local database files should not be committed to public repositories.
* Personally identifiable information should be minimized.
* Access to enrollment and attendance records should be controlled.
* Production deployments should use appropriate authentication and authorization.
* Biometric data should be encrypted where required.
* The system should be deployed in accordance with applicable institutional and privacy regulations.

> This project is intended for research, educational, and prototype purposes and should undergo appropriate security, privacy, and accuracy validation before deployment in real-world environments.

---

# ⚠️ Limitations

The current prototype may have limitations including:

* Recognition performance can vary with lighting conditions.
* Extreme face angles may reduce recognition reliability.
* Occlusions such as masks can affect recognition.
* Camera quality can influence detection performance.
* Threshold selection affects false acceptance and false rejection rates.
* SQLite is suitable for lightweight deployments but may not be ideal for large-scale concurrent systems.
* Biometric systems require careful security and privacy management.

---

# 🔮 Future Enhancements

Future versions of the system can incorporate:

## 🤖 Advanced AI

* More efficient face recognition models
* Transformer-based face recognition
* Improved anti-spoofing mechanisms
* Liveness detection
* Domain adaptation

## 📱 Deployment

* Mobile application
* Edge-device deployment
* Raspberry Pi / NVIDIA Jetson deployment
* Cloud-based institutional dashboard

## 📊 Analytics

* Attendance percentage analysis
* Student-wise reports
* Monthly and semester reports
* Automated absence detection
* Attendance trend visualization

## 🔐 Security

* Face anti-spoofing
* Liveness detection
* Encrypted biometric storage
* Role-based authentication
* Secure API communication

## 🏫 Institutional Integration

The system can potentially be integrated with:

* Student Information Systems
* College ERP platforms
* Faculty dashboards
* Academic management systems
* Notification services

---

# 👥 Contributors

## Guru Tulasi Hima Bindu

**Project Lead & AI Integration**

* Core system architecture
* Face recognition pipeline
* InsightFace integration
* AI model integration
* Backend integration
* Overall project coordination

GitHub: [@24A31A4209](https://github.com/24A31A4209)

---

## Piradi Sai Rupa Sri

**Backend & Database Management**

* Flask backend development
* API implementation
* SQLite database management
* Attendance data processing
* Backend optimization

GitHub: [@24A31A4227](https://github.com/24A31A4227)

---

## Jampa Siri Chandana Priya

**UI/UX Design & Frontend Development**

* Web interface development
* HTML/CSS implementation
* JavaScript functionality
* User experience design
* Dashboard interface

GitHub: [@24A31A4210](https://github.com/24A31A4210)

---

## Chitturi Dhanu Sree

**Data Engineering & Performance Testing**

* Dataset preparation
* Face enrollment processing
* Recognition testing
* Performance evaluation
* Test case development

GitHub: [@24A31A4207](https://github.com/24A31A4207)

---

## Chikkala Mary Blessica

**Technical Documentation & Deployment**

* Technical documentation
* Requirement analysis
* Environment configuration
* Deployment support
* Project documentation

GitHub: [@24A31A4206](https://github.com/24A31A4206)

---

# 📜 License

This project is developed for **academic and research purposes**.

If you intend to use, modify, or distribute this project commercially, please contact the project authors for appropriate licensing terms.

---

# 🙏 Acknowledgements

We acknowledge the open-source communities and research efforts behind the technologies used in this project, particularly:

* InsightFace
* OpenCV
* Flask
* SQLite
* SCRFD
* MobileFaceNet

These technologies provide the foundation for implementing efficient computer-vision and biometric applications.

---

# 📚 References

1. InsightFace — Open-source 2D/3D face analysis project.
2. SCRFD — Sample and Computation Redistribution for Efficient Face Detection.
3. MobileFaceNet — Efficient CNN architecture for face recognition.
4. OpenCV — Open-source computer vision library.
5. Flask — Lightweight Python web framework.
6. SQLite — Embedded relational database engine.

---

# ⭐ Project Summary

**Smart Attendance System** demonstrates the practical integration of:

**Artificial Intelligence + Computer Vision + Deep Learning + Facial Biometrics + Web Technologies + Database Management**

into a unified automated attendance platform.

The project provides a foundation for further research into **efficient, secure, privacy-aware, and scalable AI-powered biometric systems**.

---

## 👨‍💻 Developed by the Smart Attendance System Team

**Guru Tulasi Hima Bindu • Piradi Sai Rupa Sri • Jampa Siri Chandana Priya • Chitturi Dhanu Sree • Chikkala Mary Blessica**

⭐ If you find this project useful, consider giving the repository a star.

```
```
