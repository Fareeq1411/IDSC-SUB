# HYGD Glaucoma Detection

This project runs a local Flask web app for glaucoma detection using the Hillel Yaffe Glaucoma Dataset (HYGD).

---


If you want to **train the model yourself**, refer to:

```
training_manual.txt
```

This project already includes trained weights, so training is optional.

---

## Before You Start

## 0. Install Git (If You Don't Have It)

If you have never used Git before, you need to install it first.

### macOS

1. Install using Homebrew (recommended):

```bash
git --version || brew install git
```

2. Or download from: [https://git-scm.com/download/mac](https://git-scm.com/download/mac)

3. Verify installation:

```bash
git --version
```

---

### Windows

1. Download Git from: [https://git-scm.com/download/win](https://git-scm.com/download/win)
2. Run the installer (use default settings)
3. After installation, open Command Prompt or PowerShell
4. Verify:

```powershell
git --version
```

---

## 0.1. Clone the Repository

Since this is a public repository, you first need to clone it to your local machine.

### macOS / Linux

```bash
git clone https://github.com/Fareeq1411/IDSC-SUB.git
cd IDSC-SUB
```

### Windows

```powershell
git clone https://github.com/Fareeq1411/IDSC-SUB.git
cd IDSC-SUB
```

---

## Dataset

This project is based on the `Hillel Yaffe Glaucoma (HYGD)` dataset selected for the competition.

* Modality: retinal fundus images
* Format: `.jpg` images with `Labels.csv`
* Target task: glaucomatous optic neuropathy (GON) detection
* Extra information: includes image quality scores for quality-aware medical computer vision modeling

---

## Project Files

* `detectorUI.py`: main Flask app
* `requirements.txt`: Python dependencies
* `best.pt`: optic disc detection model
* `resnet50_glaucoma_84.pth`: glaucoma classification model

---

## 1. Install Python

Make sure Python is installed on your machine.

Recommended version: **Python 3.10 or 3.11**

### macOS

1. Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Download and install Python
3. Verify:

```bash
python3 --version
```

### Windows

1. Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Install Python
3. Enable **Add Python to PATH** (On the bottom checkbox during installation)
4. Verify:

```powershell
python --version
```

If needed:

```powershell
py --version
```

---

## 2. Open the Project Folder

```bash
cd /path/to/HYGD-Glaucoma-Detection
```

---

## 3. Create a Virtual Environment

### macOS

```bash
python3 -m venv venv
```

### Windows

```powershell
python -m venv venv
```

---

## 4. Activate the Virtual Environment

### macOS

```bash
source venv/bin/activate
```

### Windows

Command Prompt:

```cmd
venv\Scripts\activate
```

PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

If blocked:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Then retry activation.

---

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

If needed:

```bash
python -m pip install -r requirements.txt
```

---

## 6. Run the App (Port 5050)

### macOS

```bash
python3 detectorUI.py
```

### Windows

```powershell
python detectorUI.py
```

---

## 7. If Port 5050 Is Already In Use

### macOS

```bash
lsof -i :5050
kill -9 $(lsof -ti :5050)
```

### Windows

```cmd
netstat -ano | findstr :5050
taskkill /PID <PID> /F
```

---

## 8. Open the App

Go to:

```
http://127.0.0.1:5050
```

---

## 9. Test the Model

1. Open the web app
2. Click **Upload Eye Scan**
3. Select an image
4. Wait a few seconds

The system will:

* detect the optic disc
* crop the region
* classify glaucoma

Result will be displayed on screen.

---

## Notes

* Supported formats: `png`, `jpg`, `jpeg`, `bmp`
* Runs locally (no internet needed after setup)
* Do not delete:

  * `best.pt`
  * `resnet50_glaucoma_84.pth`

---

## Quick Start

### macOS

```bash
git clone https://github.com/Fareeq1411/IDSC
cd HYGD-Glaucoma-Detection
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 detectorUI.py
```

---

### Windows

```powershell
git clone https://github.com/<your-repo-link>.git
cd HYGD-Glaucoma-Detection
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python detectorUI.py
```
