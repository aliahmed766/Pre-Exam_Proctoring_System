🎯 Pre-Exam Proctoring System

Real-Time Object Detection with AI Voice Feedback

This project is a real-time pre-exam proctoring system built using YOLOv8, Streamlit, and Google Gemini AI. It detects prohibited objects through a live camera feed and provides automatic AI-generated voice warnings to help maintain a clean and fair examination environment.

🚀 Features

📷 Live Camera Object Detection (YOLOv8 – custom trained)

🧠 AI-Generated Voice Feedback using Google Gemini

🔊 Text-to-Speech Alerts via gTTS

🎯 Adjustable confidence threshold

⏱️ Configurable audio cooldown

🔐 Secure Gemini API key handling

🌐 Web-based UI powered by Streamlit + WebRTC

🧠 Detected Objects

The system is trained to detect the following objects commonly restricted during exams:

📱 Mobile Phone

🧮 Calculator

⌚ Watch

🎒 Bag

📚 Books

📓 Notebook

📄 Paper

🛠️ Tech Stack

Python 3.9+

YOLOv8 (Ultralytics)

Streamlit

streamlit-webrtc

OpenCV

Google Gemini API

gTTS

NumPy

📁 Project Structure
Pre-Exam_Proctoring_System/
│
├── app.py              # Main Streamlit application
├── best.pt             # Custom trained YOLOv8 model
├── requirements.txt    # Python dependencies
├── README.md           # Project documentation
└── .gitattributes

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/aliahmed766/Pre-Exam_Proctoring_System.git
cd Pre-Exam_Proctoring_System

2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

🔑 Gemini API Key Setup

You can provide your Google Gemini API key in two ways:

✅ Option 1: Upload a .txt file

Create a text file containing only your API key

Upload it via the sidebar in the app

✅ Option 2: Secure Input

Enter the API key directly in the password field (hidden input)

🔐 Your API key is never displayed or stored.

▶️ Run the Application
streamlit run app.py


Then open your browser at:

http://localhost:8501

📌 How It Works

Start the camera from the web interface

YOLOv8 detects restricted objects in real-time

Gemini AI generates a short polite warning message

gTTS converts it into spoken audio feedback

Audio alerts respect cooldown timing to avoid repetition

🔒 Privacy & Security

🎥 Camera feed is processed locally

🔑 API key is session-based and hidden

🧠 No video or audio data is stored

☁️ Gemini is used only for text generation

🧪 Example Voice Messages

“I see a mobile phone and books. Please remove them.”

“Detected a calculator on the desk. Kindly put it away.”

“Books and notebooks are visible. Please clear the area.”

📈 Future Improvements

Face detection & gaze tracking

Exam rule customization

Multi-language voice support

Cloud deployment

Logging & reporting system

👤 Author

Ali Ahmed
📌 GitHub: aliahmed766

⭐ Support

If you find this project useful:

🌟 Star the repository

🍴 Fork it

🐛 Open issues or feature requests
