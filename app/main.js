// Electron main process.
//
// Owns the Python sidecar: spawns it, frames its newline-delimited JSON
// output, and forwards events to the renderer. The renderer never touches the
// child process directly -- it talks over IPC through a preload bridge with
// contextIsolation on, so page code cannot reach Node APIs.

const { app, BrowserWindow, Menu, ipcMain } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const REPO_ROOT = path.join(__dirname, "..");

let win = null;
let sidecar = null;
let stdoutBuffer = "";
let requestCounter = 0;

function resolvePython() {
  // Prefer a project virtualenv, fall back to whatever is on PATH.
  const candidates = [
    path.join(REPO_ROOT, ".venv", "bin", "python"),
    path.join(REPO_ROOT, ".venv", "Scripts", "python.exe"),
    path.join(REPO_ROOT, ".venv-win", "Scripts", "python.exe"),
  ];
  for (const c of candidates) {
    if (fs.existsSync(c)) return c;
  }
  return process.platform === "win32" ? "python.exe" : "python3";
}

function resolveCheckpoint() {
  // Most-finished stage first: DPO, then SFT, then the pretrained base.
  const candidates = [
    path.join(REPO_ROOT, "runs", "dpo", "dpo.pt"),
    path.join(REPO_ROOT, "runs", "sft", "sft.pt"),
    path.join(REPO_ROOT, "runs", "base", "best.pt"),
  ];
  return candidates.find((c) => fs.existsSync(c)) || null;
}

function send(channel, payload) {
  if (win && !win.isDestroyed()) win.webContents.send(channel, payload);
}

function startSidecar() {
  const python = resolvePython();
  const ckpt = resolveCheckpoint();
  const tokenizer = path.join(REPO_ROOT, "tokenizer.json");

  if (!ckpt) {
    send("sidecar:error", {
      message:
        "No checkpoint found. Train a model first, or copy one to runs/base/best.pt.",
    });
    return;
  }
  if (!fs.existsSync(tokenizer)) {
    send("sidecar:error", { message: `No tokenizer at ${tokenizer}.` });
    return;
  }

  sidecar = spawn(
    python,
    ["-m", "server.sidecar", "--ckpt", ckpt, "--tokenizer", tokenizer],
    { cwd: REPO_ROOT, stdio: ["pipe", "pipe", "pipe"] }
  );

  sidecar.stdout.setEncoding("utf8");
  sidecar.stdout.on("data", (chunk) => {
    stdoutBuffer += chunk;
    // Frame on newlines; a partial line stays buffered until the rest arrives.
    let idx;
    while ((idx = stdoutBuffer.indexOf("\n")) >= 0) {
      const line = stdoutBuffer.slice(0, idx).trim();
      stdoutBuffer = stdoutBuffer.slice(idx + 1);
      if (!line) continue;
      try {
        send("sidecar:event", JSON.parse(line));
      } catch (e) {
        send("sidecar:error", { message: `bad JSON from sidecar: ${line.slice(0, 200)}` });
      }
    }
  });

  sidecar.stderr.setEncoding("utf8");
  sidecar.stderr.on("data", (chunk) => send("sidecar:log", { text: chunk }));

  sidecar.on("exit", (code, signal) => {
    sidecar = null;
    send("sidecar:exit", { code, signal });
  });

  sidecar.on("error", (err) => {
    send("sidecar:error", { message: `could not start Python: ${err.message}` });
  });
}

function write(obj) {
  if (!sidecar || sidecar.killed) {
    send("sidecar:error", { message: "sidecar is not running" });
    return false;
  }
  sidecar.stdin.write(JSON.stringify(obj) + "\n");
  return true;
}

function createWindow() {
  // Electron installs a stock File/Edit/View/Window menu by default. None of
  // it applies here, so it is removed rather than left as dead chrome.
  Menu.setApplicationMenu(null);

  win = new BrowserWindow({
    width: 1000,
    height: 760,
    minWidth: 560,
    backgroundColor: "#14161b",
    title: "BananaAI",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  win.loadFile(path.join(__dirname, "renderer", "index.html"));
  if (process.argv.includes("--dev")) win.webContents.openDevTools();
  win.on("closed", () => {
    win = null;
  });
}

app.whenReady().then(() => {
  createWindow();
  win.webContents.once("did-finish-load", startSidecar);

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("window-all-closed", () => {
  if (sidecar) {
    write({ type: "shutdown" });
    setTimeout(() => sidecar && sidecar.kill(), 500);
  }
  if (process.platform !== "darwin") app.quit();
});

ipcMain.handle("chat:send", (_evt, { messages, params }) => {
  const id = String(++requestCounter);
  const ok = write({ id, type: "generate", messages, params });
  return ok ? id : null;
});

ipcMain.handle("chat:cancel", () => write({ type: "cancel" }));
ipcMain.handle("chat:info", () => write({ id: "info", type: "info" }));
ipcMain.handle("chat:restart", () => {
  if (sidecar) sidecar.kill();
  stdoutBuffer = "";
  startSidecar();
  return true;
});
