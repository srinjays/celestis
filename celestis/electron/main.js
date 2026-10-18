/**
 * CELESTIS — Autonomous Satellite Mission Control
 * Inceptrix SAT · Ajinkya Singh · 2026
 * Electron Main Process
 */

const { app, BrowserWindow, shell, ipcMain } = require('electron');
const path = require('path');
const { spawn, execFile } = require('child_process');
const http = require('http');

let mainWindow = null;
let backendProc = null;

// ── Launch Python backend ─────────────────────────────────────────────────────
function startBackend() {
    const backendDir = path.join(__dirname, '..', 'backend');
    const pythonExecs = [
        'C:\\Users\\AJINKYA SINGH\\AppData\\Local\\Programs\\Python\\Python314\\python.exe',
        'python3',
        'python'
    ];
    const pythonExe = pythonExecs[0];

    console.log('[CELESTIS] Starting Python backend...');
    backendProc = spawn(
        pythonExe,
        ['-m', 'uvicorn', 'api:app', '--host', '127.0.0.1', '--port', '8001'],
        { cwd: backendDir, stdio: 'pipe' }
    );
    backendProc.stdout.on('data', d => process.stdout.write('[backend] ' + d));
    backendProc.stderr.on('data', d => process.stderr.write('[backend] ' + d));
    backendProc.on('exit', code => console.log('[CELESTIS] Backend exited:', code));
}

// ── Wait for backend health check ────────────────────────────────────────────
function waitForBackend(retries = 60) {
    return new Promise((resolve) => {
        const attempt = () => {
            http.get('http://127.0.0.1:8001/health', res => {
                if (res.statusCode === 200) { resolve(true); }
                else retry();
            }).on('error', retry);
        };
        const retry = () => {
            if (retries-- <= 0) { resolve(false); return; }
            setTimeout(attempt, 1000);
        };
        attempt();
    });
}

// ── Create Window ─────────────────────────────────────────────────────────────
async function createWindow() {
    startBackend();

    mainWindow = new BrowserWindow({
        width: 1600,
        height: 960,
        minWidth: 1280,
        minHeight: 780,
        frame: false,                    // frameless — custom title bar
        titleBarStyle: 'hidden',
        backgroundColor: '#0a0f1e',      // matches Celestis dark background
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'preload.js'),
            webSecurity: false             // allows fetch to localhost:8001
        },
        icon: path.join(__dirname, '..', 'assets', 'icon.png'),
        show: false
    });

    // Show splash then load once backend is up
    mainWindow.once('ready-to-show', () => mainWindow.show());

    console.log('[CELESTIS] Waiting for backend...');
    const ready = await waitForBackend();
    console.log('[CELESTIS] Backend ready:', ready);

    // Load the frontend
    mainWindow.loadFile(path.join(__dirname, '..', 'index.html'));

    // Open external links in system browser, not Electron
    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        shell.openExternal(url);
        return { action: 'deny' };
    });

    // Dev tools shortcut (Ctrl+Shift+I)
    mainWindow.webContents.on('before-input-event', (_, input) => {
        if (input.control && input.shift && input.key === 'I') {
            mainWindow.webContents.toggleDevTools();
        }
    });

    mainWindow.on('closed', () => { mainWindow = null; });
}

// ── Window controls via IPC ───────────────────────────────────────────────────
ipcMain.on('window-minimize', () => mainWindow?.minimize());
ipcMain.on('window-maximize', () => {
    mainWindow?.isMaximized() ? mainWindow.unmaximize() : mainWindow.maximize();
});
ipcMain.on('window-close', () => mainWindow?.close());

// ── App lifecycle ─────────────────────────────────────────────────────────────
app.setName('CELESTIS');
app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    if (backendProc) { backendProc.kill(); backendProc = null; }
    if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
    if (backendProc) { backendProc.kill(); }
});
