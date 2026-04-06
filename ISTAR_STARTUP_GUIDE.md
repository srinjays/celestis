# I-STAR Mission Control — Startup Guide

Welcome to the **I-STAR (Indian Satellite Threat Assessment & Response) Autonomous Mission Control** software. This application acts as a standalone desktop client that connects to a local Python telemetry backend.

## How to Launch the Software

You do not need to open a web browser to run this software. It is a native desktop application built with Electron and Python.

### Option 1: The One-Click Command (Recommended)

Whenever you want to start the software, simply open a **PowerShell** or **Command Prompt** terminal, navigate to the `d:\inceptrix_sat` folder, and run:

```bash
cd d:\inceptrix_sat
npm start
```

**What this does automatically:**
1. It launches a stealthy background process that starts the **Python FastAPI Server** (running the ML Models and Physics Equations).
2. It waits for the Python backend to become healthy.
3. It opens the dark-themed, frameless **I-STAR Mission Control Window** right on your desktop.

---

### Option 2: Running the Server and UI Separately (For Debugging)

If you are modifying the Python code and want to see the server logs, you can run the two components in separate terminal windows:

**Terminal 1 (The Backend Brain):**
```bash
cd d:\inceptrix_sat
.\venv\Scripts\activate
uvicorn backend.api:app --host 127.0.0.1 --port 4000
```
*Leave this running. It will print the machine learning calculations and API routes.*

**Terminal 2 (The Frontend UI):**
```bash
cd d:\inceptrix_sat
npm run electron
```
*This will open the desktop window interface.*

---

## How to Package it into a Standalone `.exe`

If you want to pack everything into a single Windows executable file (so you can send it to other ISRO team members or reviewers without them needing to install Python or Node.js), run:

```bash
cd d:\inceptrix_sat
npm run build:win
```

This will take a few minutes. Once finished, you will find an **I-STAR Setup.exe** inside the `dist/` folder.

---

## Operating the Dashboard

Once the software window is open:
1. Click the **CONNECT LIVE** button in the top right window frame.
2. The dashboard will connect to the backend simulated feed. 
3. Watch the **Atmospheric Intelligence** quadrant detect solar weather (Kp surges and density anomalies).
4. Watch the **Fleet Readiness** matrix as satellites begin to experience unmodeled drag and drift from their nominal orbits (turning amber or red).
5. Flip the switch in the **Autonomous Maneuver Command Center** to test the difference between Human-on-the-loop (manual authorization) and M2M Autonomy.
