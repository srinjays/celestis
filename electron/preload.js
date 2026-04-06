/**
 * Celestis — Electron Preload Script
 * Safely exposes IPC window controls to the renderer (index.html)
 */
const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('celestisApp', {
    minimize: () => ipcRenderer.send('window-minimize'),
    maximize: () => ipcRenderer.send('window-maximize'),
    close: () => ipcRenderer.send('window-close'),
    isElectron: true,
    version: '2.0.0'
});
