# HYGD Glaucoma Detection

This project runs a local Flask web app for glaucoma detection using the Hillel Yaffe Glaucoma Dataset (HYGD).

## Dataset

This project is based on the `Hillel Yaffe Glaucoma (HYGD)` dataset selected for the competition.

- Modality: retinal fundus images
- Format: `.jpg` images with `Labels.csv`
- Target task: glaucomatous optic neuropathy (GON) detection
- Extra information: includes image quality scores for quality-aware medical computer vision modeling

## Project Files

- `detectorUI.py`: main Flask app
- `requirements.txt`: Python dependencies
- `best.pt`: optic disc detection model
- `resnet50_glaucoma_84.pth`: glaucoma classification model

## Before You Start

Make sure Python is installed on your machine.

Recommended Python version: `Python 3.10` or `Python 3.11`

## 1. Install Python

### macOS

1. Go to [python.org/downloads](https://www.python.org/downloads/).
2. Download the latest stable Python 3 installer for macOS.
3. Run the installer.
4. Open Terminal and confirm Python is installed:

```bash
python3 --version
```

### Windows

1. Go to [python.org/downloads](https://www.python.org/downloads/).
2. Download the latest stable Python 3 installer for Windows.
3. Run the installer.
4. Very important: enable `Add Python to PATH` during installation.
5. Open Command Prompt or PowerShell and confirm Python is installed:

```powershell
python --version
```

If `python` does not work on Windows, try:

```powershell
py --version
```

## 2. Open the Project Folder

Open Terminal on macOS or Command Prompt / PowerShell on Windows, then move into the project folder.

Example:

```bash
cd /Users/fareeqshahfitri/Desktop/Self_Project/Submission_IDSC
```

## 3. Create a Virtual Environment

### macOS

```bash
python3 -m venv venv
```

### Windows

Command Prompt:

```cmd
python -m venv venv
```

If needed:

```cmd
py -m venv venv
```

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

If PowerShell blocks this script, either use Command Prompt instead or run:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

Then activate the virtual environment again:

```powershell
venv\Scripts\Activate.ps1
```

After activation, you should see `(venv)` in your terminal.

## 5. Install the Requirements

### macOS

```bash
pip install -r requirements.txt
```

If `pip` does not work:

```bash
python3 -m pip install -r requirements.txt
```

### Windows

```powershell
pip install -r requirements.txt
```

If `pip` does not work:

```powershell
python -m pip install -r requirements.txt
```

## 6. Run the App on Port 5050

This project uses `detectorUI.py` and already runs on port `5050`.

### macOS

```bash
python3 detectorUI.py
```

### Windows

```powershell
python detectorUI.py
```

If needed:

```powershell
py detectorUI.py
```

## 7. If Port 5050 Is Already In Use

Clear the process using port `5050`, then run the app again.

### macOS

Check which process is using the port:

```bash
lsof -i :5050
```

Kill the process:

```bash
kill -9 $(lsof -ti :5050)
```

Then rerun:

```bash
python3 detectorUI.py
```

### Windows

Command Prompt:

```cmd
netstat -ano | findstr :5050
```

Look at the PID in the last column, then kill it:

```cmd
taskkill /PID <PID> /F
```

Then rerun:

```cmd
python detectorUI.py
```

PowerShell alternative:

```powershell
Get-NetTCPConnection -LocalPort 5050 | Select-Object LocalAddress, LocalPort, State, OwningProcess
```

Kill the process:

```powershell
Stop-Process -Id <PID> -Force
```

Then rerun:

```powershell
python detectorUI.py
```

## 8. Open the App in Your Browser

Once the server is running, open:

```text
http://127.0.0.1:5050
```

## 9. Test the Prediction Model

1. Open `http://127.0.0.1:5050`.
2. Click `Upload Eye Scan`.
3. Select the eye scan image you want to test.
4. Wait a few seconds while the app:
   - uploads the image
   - detects the optic disc
   - crops the disc region
   - runs glaucoma analysis
5. The result will appear on the screen after processing completes.

## Notes

- Supported upload formats: `png`, `jpg`, `jpeg`, `bmp`
- The app runs locally on your own machine
- Do not delete `best.pt` or `resnet50_glaucoma_84.pth`, because the model needs both files to work

## Quick Start Summary

### macOS

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 detectorUI.py
```

Open:

```text
http://127.0.0.1:5050
```

### Windows

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python detectorUI.py
```

Open:

```text
http://127.0.0.1:5050
```
