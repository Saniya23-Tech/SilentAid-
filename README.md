# SilentAid 🚨  
AI-powered emergency detection system for hospitals – built for the Intel AI Hackathon.
## 💡 Problem Statement  
Imagine a hospital where a patient suddenly starts shivering or showing signs of stress...  
But no one is around.  
They can’t speak. They can’t call for help. Maybe they're deaf, unconscious, or too weak.  
Normally, someone has to notice and run to alert a nurse. This wastes precious time.  
## 🛠️ Our Solution — SilentAid  
SilentAid is an **AI-powered emergency camera system** that continuously monitors **registered patients**.
When it detects signs of stress, pain, shivering, or heart-touch gestures, it automatically:
- 🔊 Plays a siren  
- 💡 Glows a red alert bulb  
- 🖥️ Displays "EMERGENCY in Room 404"  
- 🎙️ (Optionally) connects to speaker systems for real-time alerts to the nurse station
## ✅ Key Feature  
**Only registered patients are monitored**, preventing false alerts or crowd misdetection.
## ⚙️ Technologies Used
- `Python`
- `OpenCV` – face and motion detection  
- `NumPy` – image processing  
- `pygame` – audio siren  
- `Haar Cascade` – facial detection
## 📁 Files in this Repo  
- `main.py` – Main detection and monitoring logic  
- `my_sound.py` – Audio playback helper using `pygame`  
- `requirements.txt` – All required Python packages  
- `README.md` – This file  
- `.gitignore`, `LICENSE`
## 🚀 How to Run
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   python main.py
 **Add your own siren sound file** named `siren.mp3` in the root directory.  
   ⚠️ *Note: Due to file restrictions, `siren.mp3` is not uploaded to this repository. You can download any siren sound of your choice and rename it to `siren.mp3`.*
Demo video of Silent Aid:https://youtube.com/shorts/iH4Pwnjrsts?si=ihQI7bhyVYqZujtH
