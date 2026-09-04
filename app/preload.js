// Context bridge: the renderer gets exactly these four calls and one event
// stream, and no access to Node, the filesystem, or the child process.

const { contextBridge, ipcRenderer } = require("electron");

contextBridge.exposeInMainWorld("bananaai", {
  send: (messages, params) => ipcRenderer.invoke("chat:send", { messages, params }),
  cancel: () => ipcRenderer.invoke("chat:cancel"),
  info: () => ipcRenderer.invoke("chat:info"),
  restart: () => ipcRenderer.invoke("chat:restart"),

  onEvent: (cb) => ipcRenderer.on("sidecar:event", (_e, payload) => cb(payload)),
  onError: (cb) => ipcRenderer.on("sidecar:error", (_e, payload) => cb(payload)),
  onLog: (cb) => ipcRenderer.on("sidecar:log", (_e, payload) => cb(payload)),
  onExit: (cb) => ipcRenderer.on("sidecar:exit", (_e, payload) => cb(payload)),
});
