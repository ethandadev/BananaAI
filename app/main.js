// Electron main process.
//
// Owns two Python sidecars and keeps them apart: an inference process that
// holds a model in memory for chat, and a training process that runs the
// pipeline. They are separate because training saturates the GPU, and sharing
// one process would make the chat window hang for hours.
//
// The renderer never touches either child directly. It talks over IPC through
// a preload bridge with contextIsolation on, so page code cannot reach Node.

const { app, BrowserWindow, Menu, ipcMain, dialog } = require("electron");
const { spawn } = require("child_process");
const path = require("path");
const fs = require("fs");

const REPO_ROOT = path.join(__dirname, "..");

let win = null;
let requestCounter = 0;

// Each entry holds a child process and the partial line left over from the
// last stdout chunk. JSON arrives newline-delimited but chunks split anywhere.
const sidecars = {
  chat: { proc: null, buffer: "", args: null },
  train: { proc: null, buffer: "", args: null },
};

function resolvePython() {
  const candidates = [
    path.join(REPO_ROOT, ".venv", "bin", "python"),
    path.join(REPO_ROOT, ".venv", "Scripts", "python.exe"),
    path.join(REPO_ROOT, ".venv-win", "Scripts", "python.exe"),
  ];
  for (const c of candidates) if (fs.existsSync(c)) return c;
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

function startSidecar(which, moduleArgs) {
  const entry = sidecars[which];
  if (entry.proc) return true;

  const python = resolvePython();
  entry.args = moduleArgs;
  entry.buffer = "";

  const child = spawn(python, moduleArgs, {
    cwd: REPO_ROOT,
    stdio: ["pipe", "pipe", "pipe"],
  });
  entry.proc = child;

  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk) => {
    entry.buffer += chunk;
    let idx;
    while ((idx = entry.buffer.indexOf("\n")) >= 0) {
      const line = entry.buffer.slice(0, idx).trim();
      entry.buffer = entry.buffer.slice(idx + 1);
      if (!line) continue;
      try {
        send(`${which}:event`, JSON.parse(line));
      } catch {
        send(`${which}:error`, {
          message: `unreadable output from the ${which} process: ${line.slice(0, 200)}`,
        });
      }
    }
  });

  child.stderr.setEncoding("utf8");
  child.stderr.on("data", (chunk) => send(`${which}:log`, { text: chunk }));

  child.on("exit", (code, signal) => {
    entry.proc = null;
    send(`${which}:exit`, { code, signal });
  });

  child.on("error", (err) => {
    entry.proc = null;
    send(`${which}:error`, { message: `could not start Python: ${err.message}` });
  });

  return true;
}

function write(which, obj) {
  const entry = sidecars[which];
  if (!entry.proc || entry.proc.killed) {
    send(`${which}:error`, { message: `the ${which} process is not running` });
    return false;
  }
  entry.proc.stdin.write(JSON.stringify(obj) + "\n");
  return true;
}

function stopSidecar(which, { force = false } = {}) {
  const entry = sidecars[which];
  if (!entry.proc) return;
  write(which, { type: "shutdown" });
  const child = entry.proc;
  setTimeout(() => {
    if (child && !child.killed) child.kill(force ? "SIGKILL" : "SIGTERM");
  }, force ? 200 : 3000);
}

// --- chat ------------------------------------------------------------------

function startChat() {
  const ckpt = resolveCheckpoint();
  const tokenizer = path.join(REPO_ROOT, "tokenizer.json");

  if (!ckpt) {
    send("chat:error", {
      message: "No trained model yet. Train one on the Train tab first.",
      code: "no_checkpoint",
    });
    return;
  }
  if (!fs.existsSync(tokenizer)) {
    send("chat:error", { message: "No tokenizer yet — train a model first.", code: "no_tokenizer" });
    return;
  }
  startSidecar("chat", ["-m", "server.sidecar", "--ckpt", ckpt, "--tokenizer", tokenizer]);
}

// --- window ----------------------------------------------------------------

function createWindow() {
  // Electron installs a stock File/Edit/View/Window menu by default. None of
  // it applies here, so it is removed rather than left as dead chrome.
  Menu.setApplicationMenu(null);

  win = new BrowserWindow({
    width: 1180,
    height: 820,
    minWidth: 720,
    minHeight: 560,
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
  win.webContents.once("did-finish-load", () => {
    startSidecar("train", ["-m", "server.trainer"]);
    startChat();
  });

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on("before-quit", (event) => {
  // A run in progress checkpoints on shutdown; give it a moment rather than
  // killing it and discarding the work.
  const training = sidecars.train.proc;
  if (training && !app.quittingConfirmed) {
    event.preventDefault();
    app.quittingConfirmed = true;
    stopSidecar("train");
    stopSidecar("chat");
    setTimeout(() => app.quit(), 2500);
  }
});

app.on("window-all-closed", () => {
  stopSidecar("chat");
  if (process.platform !== "darwin") app.quit();
});

// --- IPC -------------------------------------------------------------------

ipcMain.handle("chat:send", (_evt, { messages, params }) => {
  const id = String(++requestCounter);
  return write("chat", { id, type: "generate", messages, params }) ? id : null;
});
ipcMain.handle("chat:cancel", () => write("chat", { type: "cancel" }));
ipcMain.handle("chat:info", () => write("chat", { id: "info", type: "info" }));
ipcMain.handle("chat:restart", () => {
  stopSidecar("chat", { force: true });
  setTimeout(startChat, 400);
  return true;
});

ipcMain.handle("train:start", (_evt, { stages }) =>
  write("train", { id: String(++requestCounter), type: "start", stages }));
ipcMain.handle("train:stop", () => write("train", { type: "stop" }));
ipcMain.handle("train:plan", () => write("train", { id: "plan", type: "plan" }));
ipcMain.handle("train:status", () => write("train", { id: "status", type: "status" }));

ipcMain.handle("dialog:pickFolder", async () => {
  const result = await dialog.showOpenDialog(win, {
    title: "Choose a folder of documents to train on",
    properties: ["openDirectory"],
  });
  return result.canceled ? null : result.filePaths[0];
});
