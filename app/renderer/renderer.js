// Renderer: two views over two sidecars.
//
// All model and log text is inserted with textContent. Model output is
// untrusted -- a generated <img onerror> must never execute inside the app --
// and so is a Python traceback.

const $ = (id) => document.getElementById(id);

const toast = $("toast");
const statusDot = $("status-dot");
const headerMeta = $("header-meta");

function showToast(text, ms = 6000) {
  toast.textContent = text;
  toast.hidden = false;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => { toast.hidden = true; }, ms);
}

function setStatus(state, text) {
  statusDot.dataset.state = state;
  if (text) headerMeta.textContent = text;
}

// --------------------------------------------------------------- tabs

const views = { train: $("view-train"), chat: $("view-chat") };
const tabs = { train: $("tab-train"), chat: $("tab-chat") };

function showTab(name) {
  for (const [key, view] of Object.entries(views)) {
    view.hidden = key !== name;
    tabs[key].setAttribute("aria-selected", String(key === name));
  }
  if (name === "train") drawChart();
  if (name === "chat" && chatReady) input.focus();
}
tabs.train.addEventListener("click", () => showTab("train"));
tabs.chat.addEventListener("click", () => showTab("chat"));

// --------------------------------------------------------------- training

const startBtn = $("start-train");
const stopBtn = $("stop-train");
const stageList = $("stage-list");
const progressTitle = $("progress-title");
const progressStage = $("progress-stage");
const progressFill = $("progress-fill");
const logEl = $("train-log");
const canvas = $("loss-chart");

let losses = [];        // {step, loss}
let valLosses = [];     // {step, loss}
let running = false;
let logLines = [];

function log(text) {
  logLines.push(text);
  if (logLines.length > 400) logLines = logLines.slice(-400);
  logEl.textContent = logLines.join("\n");
  logEl.scrollTop = logEl.scrollHeight;
}

function formatDuration(seconds) {
  if (!isFinite(seconds) || seconds < 0) return "—";
  if (seconds < 90) return `${Math.round(seconds)}s`;
  const minutes = seconds / 60;
  if (minutes < 90) return `${minutes.toFixed(0)}m`;
  const hours = minutes / 60;
  if (hours < 48) return `${hours.toFixed(1)}h`;
  return `${(hours / 24).toFixed(1)} days`;
}

function selectedStages() {
  return [...stageList.querySelectorAll("input:checked")].map((el) => el.value);
}

function markStage(name, status) {
  const box = stageList.querySelector(`input[value="${name}"]`);
  if (box) box.closest("li").dataset.status = status;
}

function setRunning(on) {
  running = on;
  startBtn.hidden = on;
  stopBtn.hidden = !on;
  stageList.querySelectorAll("input").forEach((el) => { el.disabled = on; });
}

startBtn.addEventListener("click", async () => {
  const stages = selectedStages();
  if (!stages.length) {
    showToast("Pick at least one thing to run.");
    return;
  }
  losses = [];
  valLosses = [];
  logLines = [];
  stageList.querySelectorAll("li").forEach((li) => delete li.dataset.status);
  setRunning(true);
  progressTitle.textContent = "Starting…";
  log(`starting: ${stages.join(" → ")}`);
  await window.bananaai.train.start(stages);
});

stopBtn.addEventListener("click", async () => {
  stopBtn.disabled = true;
  progressTitle.textContent = "Finishing this step, then saving…";
  log("stop requested — the current step will finish and checkpoint");
  await window.bananaai.train.stop();
});

$("pick-folder").addEventListener("click", async () => {
  const folder = await window.bananaai.pickFolder();
  if (!folder) return;
  $("folder-path").textContent = folder;
  showToast(
    "Set data.custom_dir in bananaai.toml to this path, then run the data stage. " +
    "A personal folder is usually far too small to pretrain on alone — the data " +
    "step will tell you how far it goes.",
    12000
  );
});

window.bananaai.train.onEvent((event) => {
  switch (event.type) {
    case "ready": {
      if (event.problems && event.problems.length) {
        event.problems.forEach((p) => log(`config problem: ${p}`));
        showToast(`Configuration problem: ${event.problems[0]}`, 12000);
      }
      if (event.device) {
        $("fact-device").textContent = event.device;
        $("fact-preset").textContent = event.preset ?? "—";
        $("fact-params").textContent = event.parameters
          ? `${(event.parameters / 1e6).toFixed(0)}M` : "—";
        $("fact-hours").textContent = event.estimated_hours
          ? formatDuration(event.estimated_hours * 3600) : "—";
        const warnings = $("machine-warnings");
        warnings.textContent = "";
        (event.plan_warnings || []).forEach((w) => {
          const li = document.createElement("li");
          li.textContent = w;
          warnings.append(li);
        });
        setStatus("ready", `${event.preset} · ${event.device.split(",")[0]}`);
      }
      break;
    }
    case "plan":
      if (event.summary) log(event.summary);
      break;
    case "stage":
      markStage(event.name, event.status);
      progressStage.textContent = `${event.name} — ${event.status}`;
      log(`[${event.name}] ${event.status}`);
      if (event.status === "done" && event.detail) {
        log(`  ${JSON.stringify(event.detail).slice(0, 300)}`);
      }
      break;
    case "note":
    case "advice":
      log(event.text);
      break;
    case "setup":
      progressTitle.textContent = "Training";
      log(`${event.parameters.toLocaleString()} parameters, ${event.precision}, `
        + `${event.total_steps.toLocaleString()} steps on ${event.device}`);
      break;
    case "progress": {
      const pct = ((event.step + 1) / event.total_steps) * 100;
      progressFill.style.width = `${Math.min(100, pct)}%`;
      progressTitle.textContent = `Training — ${pct.toFixed(1)}%`;
      $("read-step").textContent =
        `${event.step.toLocaleString()} / ${event.total_steps.toLocaleString()}`;
      $("read-loss").textContent = event.loss.toFixed(4);
      $("read-speed").textContent =
        `${(event.tokens_per_second / 1000).toFixed(1)}k tok/s`;
      $("read-eta").textContent = formatDuration(event.eta_seconds);
      losses.push({ step: event.step, loss: event.loss });
      drawChart();
      break;
    }
    case "eval":
      valLosses.push({ step: event.step, loss: event.val_loss });
      $("read-val").textContent = event.val_loss.toFixed(4)
        + (event.best ? "  (best)" : "");
      drawChart();
      break;
    case "checkpoint":
      log(`saved checkpoint at step ${event.step}${event.best ? " (best so far)" : ""}`);
      break;
    case "sample":
      log(`sample: ${event.text}`);
      break;
    case "aborted":
      setRunning(false);
      stopBtn.disabled = false;
      progressTitle.textContent = "Training stopped";
      showToast(event.reason, 15000);
      log(`ABORTED: ${event.reason}`);
      break;
    case "stopped":
      log(`stopped at step ${event.step}, checkpoint written`);
      break;
    case "finished":
      setRunning(false);
      stopBtn.disabled = false;
      progressTitle.textContent = event.stopped ? "Stopped" : "Finished";
      log(`finished in ${formatDuration(event.seconds)}`);
      if (!event.stopped) {
        showToast("Training finished. Switch to the Chat tab to try it.", 12000);
        window.bananaai.chat.restart();
      }
      break;
    case "error":
      setRunning(false);
      stopBtn.disabled = false;
      progressTitle.textContent = "Failed";
      showToast(event.message, 15000);
      log(`ERROR${event.stage ? ` in ${event.stage}` : ""}: ${event.message}`);
      if (event.traceback) log(event.traceback);
      break;
  }
});

window.bananaai.train.onError(({ message }) => {
  setRunning(false);
  showToast(message, 12000);
  log(`ERROR: ${message}`);
});

window.bananaai.train.onLog(({ text }) => {
  text.split("\n").forEach((line) => {
    if (line.trim()) log(line);
  });
});

window.bananaai.train.onExit(({ code }) => {
  setRunning(false);
  if (code !== 0 && code !== null) log(`training process exited with code ${code}`);
});

// --------------------------------------------------------------- loss chart

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function drawChart() {
  const ctx = canvas.getContext("2d");
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || canvas.width;
  const height = 260;
  if (canvas.width !== width * ratio) {
    canvas.width = width * ratio;
    canvas.height = height * ratio;
  }
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);

  if (losses.length < 2) {
    ctx.fillStyle = cssVar("--muted");
    ctx.font = "13px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.fillText("waiting for the first steps…", width / 2, height / 2);
    return;
  }

  const pad = { left: 52, right: 14, top: 14, bottom: 26 };
  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;

  const all = losses.concat(valLosses);
  const maxStep = Math.max(...all.map((p) => p.step), 1);
  let lo = Math.min(...all.map((p) => p.loss));
  let hi = Math.max(...all.map((p) => p.loss));
  if (hi - lo < 1e-6) { hi += 0.5; lo -= 0.5; }
  const padY = (hi - lo) * 0.08;
  lo -= padY; hi += padY;

  const x = (step) => pad.left + (step / maxStep) * plotW;
  const y = (loss) => pad.top + (1 - (loss - lo) / (hi - lo)) * plotH;

  // grid and axis labels
  ctx.strokeStyle = cssVar("--line");
  ctx.fillStyle = cssVar("--muted");
  ctx.lineWidth = 1;
  ctx.font = "11px ui-monospace, monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const value = lo + ((hi - lo) * i) / 4;
    const py = Math.round(y(value)) + 0.5;
    ctx.beginPath();
    ctx.moveTo(pad.left, py);
    ctx.lineTo(width - pad.right, py);
    ctx.stroke();
    ctx.fillText(value.toFixed(2), pad.left - 8, py);
  }

  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillText("0", pad.left, height - pad.bottom + 8);
  ctx.fillText(maxStep.toLocaleString(), width - pad.right, height - pad.bottom + 8);

  const line = (points, colour, dashed) => {
    if (points.length < 2) return;
    ctx.strokeStyle = colour;
    ctx.lineWidth = 2;
    ctx.setLineDash(dashed ? [5, 4] : []);
    ctx.beginPath();
    points.forEach((p, i) => {
      const px = x(p.step);
      const py = y(p.loss);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    });
    ctx.stroke();
    ctx.setLineDash([]);
  };

  line(losses, cssVar("--accent"), false);
  line(valLosses, cssVar("--good"), true);

  // emphasise the latest point, which is the one being watched
  const last = losses[losses.length - 1];
  ctx.fillStyle = cssVar("--accent");
  ctx.beginPath();
  ctx.arc(x(last.step), y(last.loss), 3.5, 0, Math.PI * 2);
  ctx.fill();

  $("chart-caption").textContent = valLosses.length
    ? "Solid: training loss. Dashed: validation loss. Both should fall, and validation is the honest one."
    : "Training loss. It should fall steeply at first, then flatten.";
}

window.addEventListener("resize", () => { if (!views.train.hidden) drawChart(); });

// --------------------------------------------------------------- chat

const transcript = $("transcript");
const emptyState = $("empty-state");
const input = $("input");
const sendBtn = $("send");
const chatStopBtn = $("stop");
const temperature = $("temperature");
const temperatureValue = $("temperature-value");
const maxTokens = $("max-tokens");

let messages = [];
let activeId = null;
let activeBubble = null;
let chatReady = false;

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
  transcript.scrollTop = transcript.scrollHeight;
  return body;
}

function setGenerating(on) {
  sendBtn.hidden = on;
  chatStopBtn.hidden = !on;
  input.disabled = on;
  if (!on) input.focus();
}

async function send() {
  const text = input.value.trim();
  if (!text || activeId || !chatReady) return;

  messages.push({ role: "user", content: text });
  addBubble("user", text);
  input.value = "";
  autoGrow();

  activeBubble = addBubble("assistant", "");
  activeBubble.classList.add("streaming");
  setGenerating(true);

  const id = await window.bananaai.chat.send(messages, {
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
  if (event && event.tokens > 0) {
    setStatus("ready", `${event.tokens} tokens · ${event.tokens_per_second.toFixed(1)} tok/s`);
  }
  activeId = null;
  activeBubble = null;
  setGenerating(false);
}

window.bananaai.chat.onEvent((event) => {
  switch (event.type) {
    case "ready":
      chatReady = true;
      setStatus("ready",
        `${(event.parameters / 1e6).toFixed(0)}M · ${event.stage} · ${event.device}`);
      break;
    case "token":
      if (activeBubble && event.id === activeId) {
        activeBubble.textContent += event.text;
        transcript.scrollTop = transcript.scrollHeight;
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

window.bananaai.chat.onError((payload) => {
  chatReady = false;
  if (payload.code === "no_checkpoint" || payload.code === "no_tokenizer") {
    emptyState.querySelector("h1").textContent = "No model yet";
    emptyState.querySelector("p").textContent =
      "Train one on the Train tab, then come back here.";
    setStatus("loading", "no model yet");
    return;
  }
  setStatus("error", "model unavailable");
  showToast(payload.message, 12000);
  setGenerating(false);
});

window.bananaai.chat.onExit(() => {
  chatReady = false;
});

sendBtn.addEventListener("click", send);
chatStopBtn.addEventListener("click", () => window.bananaai.chat.cancel());
$("clear").addEventListener("click", () => {
  messages = [];
  transcript.querySelectorAll(".turn").forEach((n) => n.remove());
  emptyState.hidden = false;
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

drawChart();
