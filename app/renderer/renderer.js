// Renderer: transcript state and the token stream.
//
// Text is inserted with textContent only. Model output is untrusted text and
// must never be parsed as HTML -- a generated <img onerror> would otherwise
// execute inside the app.

const transcript = document.getElementById("transcript");
const emptyState = document.getElementById("empty-state");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send");
const stopBtn = document.getElementById("stop");
const clearBtn = document.getElementById("clear");
const statusDot = document.getElementById("status-dot");
const modelMeta = document.getElementById("model-meta");
const temperature = document.getElementById("temperature");
const temperatureValue = document.getElementById("temperature-value");
const maxTokens = document.getElementById("max-tokens");
const toast = document.getElementById("toast");

let messages = [];        // the conversation sent to the model
let activeId = null;      // request id currently streaming
let activeBubble = null;  // the DOM node it is writing into
let ready = false;

function setStatus(state, text) {
  statusDot.dataset.state = state;
  if (text) modelMeta.textContent = text;
}

function showToast(text, ms = 5000) {
  toast.textContent = text;
  toast.hidden = false;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => {
    toast.hidden = true;
  }, ms);
}

function addBubble(role, text = "") {
  emptyState.hidden = true;
  const wrap = document.createElement("div");
  wrap.className = `turn ${role}`;

  const label = document.createElement("div");
  label.className = "role";
  label.textContent = role === "user" ? "You" : "Model";

  const body = document.createElement("div");
  body.className = "body";
  body.textContent = text;

  wrap.append(label, body);
  transcript.append(wrap);
  scrollToEnd();
  return body;
}

function scrollToEnd() {
  transcript.scrollTop = transcript.scrollHeight;
}

function setGenerating(on) {
  sendBtn.hidden = on;
  stopBtn.hidden = !on;
  input.disabled = on;
  if (!on) input.focus();
}

async function send() {
  const text = input.value.trim();
  if (!text || activeId || !ready) return;

  messages.push({ role: "user", content: text });
  addBubble("user", text);
  input.value = "";
  autoGrow();

  activeBubble = addBubble("assistant", "");
  activeBubble.classList.add("streaming");
  setGenerating(true);

  const id = await window.bananaai.send(messages, {
    temperature: Number(temperature.value),
    max_new_tokens: Number(maxTokens.value),
    top_p: 0.95,
    top_k: 50,
    repetition_penalty: 1.1,
  });

  if (id === null) {
    activeBubble.textContent = "The model process is not running.";
    activeBubble.classList.remove("streaming");
    setGenerating(false);
    return;
  }
  activeId = id;
}

function finish(event) {
  if (activeBubble) {
    activeBubble.classList.remove("streaming");
    const content = activeBubble.textContent.trim();
    if (content) messages.push({ role: "assistant", content });
    else activeBubble.textContent = "(no output)";
  }
  if (event && typeof event.tokens_per_second === "number" && event.tokens > 0) {
    setStatus("ready", `${event.tokens} tokens · ${event.tokens_per_second.toFixed(1)} tok/s`);
  }
  activeId = null;
  activeBubble = null;
  setGenerating(false);
}

window.bananaai.onEvent((event) => {
  switch (event.type) {
    case "ready": {
      ready = true;
      const params = (event.parameters / 1e6).toFixed(0);
      setStatus("ready", `${params}M · ${event.stage} · ${event.device}`);
      input.focus();
      break;
    }
    case "info":
      setStatus("ready", `${(event.parameters / 1e6).toFixed(0)}M · ${event.stage}`);
      break;
    case "token":
      if (activeBubble && event.id === activeId) {
        activeBubble.textContent += event.text;
        scrollToEnd();
      }
      break;
    case "done":
      if (event.id === activeId) finish(event);
      break;
    case "error":
      showToast(event.message);
      finish(null);
      break;
  }
});

window.bananaai.onError((payload) => {
  setStatus("error", "model unavailable");
  showToast(payload.message, 12000);
  setGenerating(false);
});

window.bananaai.onExit(({ code }) => {
  ready = false;
  setStatus("error", `model process exited (${code})`);
  setGenerating(false);
});

// stderr is where Python tracebacks land; surface the first line rather than
// leaving the user with a silent failure.
window.bananaai.onLog(({ text }) => {
  const line = text.trim().split("\n").pop();
  if (line && /error|traceback|exception/i.test(line)) showToast(line, 10000);
});

sendBtn.addEventListener("click", send);
stopBtn.addEventListener("click", () => window.bananaai.cancel());
clearBtn.addEventListener("click", () => {
  messages = [];
  transcript.querySelectorAll(".turn").forEach((n) => n.remove());
  emptyState.hidden = false;
  setStatus(ready ? "ready" : "loading");
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    send();
  }
});

function autoGrow() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, 200)}px`;
}
input.addEventListener("input", autoGrow);

temperature.addEventListener("input", () => {
  temperatureValue.textContent = Number(temperature.value).toFixed(2);
});
