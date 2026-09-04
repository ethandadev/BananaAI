// Context bridge. The renderer gets exactly these calls and event streams,
// and no access to Node, the filesystem, or either child process.

const { contextBridge, ipcRenderer } = require("electron");

function subscribe(channel) {
  return (callback) => ipcRenderer.on(channel, (_event, payload) => callback(payload));
}

contextBridge.exposeInMainWorld("bananaai", {
  chat: {
    send: (messages, params) => ipcRenderer.invoke("chat:send", { messages, params }),
    cancel: () => ipcRenderer.invoke("chat:cancel"),
    info: () => ipcRenderer.invoke("chat:info"),
    restart: () => ipcRenderer.invoke("chat:restart"),
    onEvent: subscribe("chat:event"),
    onError: subscribe("chat:error"),
    onLog: subscribe("chat:log"),
    onExit: subscribe("chat:exit"),
  },
  train: {
    start: (stages) => ipcRenderer.invoke("train:start", { stages }),
    stop: () => ipcRenderer.invoke("train:stop"),
    plan: () => ipcRenderer.invoke("train:plan"),
    status: () => ipcRenderer.invoke("train:status"),
    onEvent: subscribe("train:event"),
    onError: subscribe("train:error"),
    onLog: subscribe("train:log"),
    onExit: subscribe("train:exit"),
  },
  pickFolder: () => ipcRenderer.invoke("dialog:pickFolder"),
});
