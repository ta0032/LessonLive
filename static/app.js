/**
 * LessonLive — app.js
 * Клиентская логика главной страницы.
 * Управляет WebSocket соединением, обновлением UI и локализацией.
 */

// ─── i18n ─────────────────────────────────────────────────────────────────────

const TRANSLATIONS = {
  ru: {
    "header.title":        "LessonLive",
    "header.idle":         "Ожидание",
    "header.recording":    "Запись идёт",
    "header.loading":      "Загрузка моделей...",
    "header.ready":        "Готово",
    "sidebar.speakers":    "Спикеры",
    "sidebar.sessions":    "Прошлые занятия",
    "speaker.save":        "Сохранить",
    "speaker.0.default":   "Преподаватель",
    "speaker.1.default":   "Ученик",
    "btn.start":           "▶ Начать запись",
    "btn.stop":            "⏹ Остановить",
    "btn.clear":           "Очистить",
    "empty.title":         "Ждём начала занятия",
    "empty.desc":          "Нажмите «Начать запись» чтобы запустить транскрипцию",
    "phrases":             "фраз",
    "sessions.empty":      "Нет сохранённых занятий",
    "toast.started":       "Запись запущена",
    "toast.stopped":       "Запись остановлена",
    "toast.saved":         "Имена сохранены",
    "toast.ws_error":      "Ошибка соединения. Переподключение...",
    "toast.ws_reconnect":  "Соединение восстановлено",
    "models.loading":      "⏳ Модели загружаются...",
    "models.ready":        "✓ Готово к работе",
    "models.no_diarize":   "⚠ Без разделения спикеров",
  },
  en: {
    "header.title":        "LessonLive",
    "header.idle":         "Idle",
    "header.recording":    "Recording",
    "header.loading":      "Loading models...",
    "header.ready":        "Ready",
    "sidebar.speakers":    "Speakers",
    "sidebar.sessions":    "Past sessions",
    "speaker.save":        "Save",
    "speaker.0.default":   "Teacher",
    "speaker.1.default":   "Student",
    "btn.start":           "▶ Start recording",
    "btn.stop":            "⏹ Stop",
    "btn.clear":           "Clear",
    "empty.title":         "Waiting to start",
    "empty.desc":          "Click 'Start recording' to begin transcription",
    "phrases":             "phrases",
    "sessions.empty":      "No saved sessions",
    "toast.started":       "Recording started",
    "toast.stopped":       "Recording stopped",
    "toast.saved":         "Names saved",
    "toast.ws_error":      "Connection error. Reconnecting...",
    "toast.ws_reconnect":  "Connection restored",
    "models.loading":      "⏳ Loading models...",
    "models.ready":        "✓ Ready",
    "models.no_diarize":   "⚠ No speaker separation",
  }
};

let currentLang = localStorage.getItem("ll_lang") || "ru";

/**
 * Возвращает перевод строки по ключу для текущего языка.
 * @param {string} key
 * @returns {string}
 */
function t(key) {
  return TRANSLATIONS[currentLang]?.[key] || TRANSLATIONS["ru"][key] || key;
}

/**
 * Применяет переводы ко всем элементам с атрибутом data-i18n.
 * Также обновляет placeholder и value кнопок.
 */
function applyTranslations() {
  document.querySelectorAll("[data-i18n]").forEach(el => {
    const key = el.getAttribute("data-i18n");
    el.textContent = t(key);
  });
  document.querySelectorAll("[data-i18n-placeholder]").forEach(el => {
    el.placeholder = t(el.getAttribute("data-i18n-placeholder"));
  });
}

/**
 * Переключает язык интерфейса.
 * Сохраняет выбор в localStorage и обновляет все тексты.
 * @param {string} lang - 'ru' или 'en'
 */
function switchLanguage(lang) {
  currentLang = lang;
  localStorage.setItem("ll_lang", lang);

  document.querySelectorAll(".ui-lang-switch button").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.lang === lang);
  });

  applyTranslations();
  updateRecordingUI(appState.isRecording);
  updateModelStatus(appState.modelLoaded, appState.pyannoteLoaded);
}


// ─── Состояние приложения ─────────────────────────────────────────────────────

const appState = {
  isRecording:    false,
  modelLoaded:    false,
  pyannoteLoaded: false,
  phraseCount:    0,
  sessionFile:    null,
  speakerNames: {
    speaker_0: null,  // null = использовать дефолтное имя
    speaker_1: null,
  }
};


// ─── WebSocket ────────────────────────────────────────────────────────────────

let ws = null;
let wsReconnectTimer = null;
let wsReconnectAttempts = 0;

/**
 * Инициализирует WebSocket соединение с сервером.
 * При успешном подключении — сбрасывает счётчик переподключений.
 * При обрыве — пробует переподключиться через нарастающий интервал.
 */
function initWebSocket() {
  if (ws && ws.readyState === WebSocket.OPEN) return;

  ws = new WebSocket(`ws://${location.host}/ws`);

  ws.onopen = () => {
    console.log("[WS] Подключён");
    if (wsReconnectAttempts > 0) {
      showToast(t("toast.ws_reconnect"), "success");
    }
    wsReconnectAttempts = 0;
    clearTimeout(wsReconnectTimer);
  };

  ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      onMessage(msg);
    } catch (e) {
      console.error("[WS] Ошибка парсинга:", e);
    }
  };

  ws.onclose = () => {
    console.log("[WS] Отключён");
    scheduleReconnect();
  };

  ws.onerror = (e) => {
    console.error("[WS] Ошибка:", e);
  };
}

/**
 * Планирует переподключение с нарастающим интервалом.
 * Интервал: 2с → 4с → 8с → макс. 15с.
 */
function scheduleReconnect() {
  wsReconnectAttempts++;
  const delay = Math.min(2000 * wsReconnectAttempts, 15000);
  showToast(t("toast.ws_error"), "error");
  wsReconnectTimer = setTimeout(initWebSocket, delay);
}

/**
 * Отправляет JSON сообщение на сервер через WebSocket.
 * @param {object} data
 */
function wsSend(data) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify(data));
  }
}


// ─── Обработка сообщений от сервера ──────────────────────────────────────────

/**
 * Главный обработчик входящих WebSocket сообщений.
 * Маршрутизирует по полю type.
 * @param {object} msg - Распарсенный JSON объект
 */
function onMessage(msg) {
  switch (msg.type) {
    case "status":
      onStatusUpdate(msg);
      break;
    case "history":
      onHistoryReceived(msg.phrases);
      break;
    case "phrase":
      onPhraseReceived(msg);
      break;
    case "error":
      showToast(msg.message, "error");
      break;
    default:
      console.warn("[WS] Неизвестный тип сообщения:", msg.type);
  }
}

/**
 * Обрабатывает обновление статуса от сервера.
 * Синхронизирует локальное состояние с состоянием сервера.
 * @param {object} msg
 */
function onStatusUpdate(msg) {
  appState.isRecording    = msg.is_recording ?? appState.isRecording;
  appState.phraseCount    = msg.phrase_count ?? appState.phraseCount;
  appState.sessionFile    = msg.session_file ?? appState.sessionFile;
  appState.modelLoaded    = msg.model_loaded ?? appState.modelLoaded;
  appState.pyannoteLoaded = msg.pyannote_loaded ?? appState.pyannoteLoaded;

  updateRecordingUI(appState.isRecording);
  updatePhraseCount(appState.phraseCount);
  updateModelStatus(appState.modelLoaded, appState.pyannoteLoaded);
}

/**
 * Получает историю текущей сессии при подключении.
 * Отображает все накопленные фразы.
 * @param {Array} phrases
 */
function onHistoryReceived(phrases) {
  clearTranscript();
  phrases.forEach(phrase => appendPhrase(phrase));
}

/**
 * Получает новую фразу в реальном времени и добавляет в UI.
 * @param {object} phrase
 */
function onPhraseReceived(phrase) {
  appendPhrase(phrase);
  appState.phraseCount++;
  updatePhraseCount(appState.phraseCount);
}


// ─── Управление записью ───────────────────────────────────────────────────────

/**
 * Запускает запись — отправляет команду start на сервер.
 * Кнопка блокируется до получения подтверждения от сервера.
 */
function startRecording() {
  wsSend({ action: "start" });
}

/**
 * Останавливает запись — отправляет команду stop на сервер.
 */
function stopRecording() {
  wsSend({ action: "stop" });
}

/**
 * Переключает запись: если идёт — останавливает, если нет — запускает.
 * Привязан к главной кнопке.
 */
function toggleRecording() {
  if (appState.isRecording) {
    stopRecording();
  } else {
    startRecording();
  }
}


// ─── Имена спикеров ───────────────────────────────────────────────────────────

/**
 * Сохраняет имена спикеров из полей ввода.
 * Отправляет команду set_names на сервер через WebSocket.
 * Также сохраняет в localStorage для восстановления после перезагрузки.
 */
function saveSpeakerNames() {
  const name0 = document.getElementById("speaker-0-name")?.value.trim();
  const name1 = document.getElementById("speaker-1-name")?.value.trim();

  const payload = { action: "set_names" };
  if (name0) payload.speaker_0 = name0;
  if (name1) payload.speaker_1 = name1;

  wsSend(payload);

  // Сохраняем локально
  if (name0) { appState.speakerNames.speaker_0 = name0; localStorage.setItem("ll_speaker_0", name0); }
  if (name1) { appState.speakerNames.speaker_1 = name1; localStorage.setItem("ll_speaker_1", name1); }

  showToast(t("toast.saved"), "success");
}

/**
 * Загружает сохранённые имена спикеров из localStorage
 * и заполняет поля ввода.
 */
function loadSpeakerNames() {
  const name0 = localStorage.getItem("ll_speaker_0");
  const name1 = localStorage.getItem("ll_speaker_1");

  const input0 = document.getElementById("speaker-0-name");
  const input1 = document.getElementById("speaker-1-name");

  if (input0 && name0) { input0.value = name0; appState.speakerNames.speaker_0 = name0; }
  if (input1 && name1) { input1.value = name1; appState.speakerNames.speaker_1 = name1; }
}


// ─── Отображение транскрипции ─────────────────────────────────────────────────

/**
 * Добавляет новую фразу в область транскрипции.
 * Создаёт DOM-элемент с цветовой кодировкой по спикеру.
 * Прокручивает область вниз после добавления.
 * @param {object} phrase - {speaker_id, speaker_name, text, language, timestamp}
 */
function appendPhrase(phrase) {
  const container = document.getElementById("transcript-scroll");
  const emptyState = document.getElementById("transcript-empty");

  // Скрываем пустое состояние
  if (emptyState) emptyState.style.display = "none";

  // Создаём элемент фразы
  const el = document.createElement("div");
  el.className = `phrase ${phrase.speaker_id || "speaker-unknown"}`;
  el.dataset.index = phrase.index;

  const langClass = `lang-${phrase.language || "unknown"}`;
  const langLabel = (phrase.language || "??").toUpperCase();
  const speakerName = phrase.speaker_name || t(`speaker.${phrase.speaker_id === "speaker_1" ? "1" : "0"}.default`);

  el.innerHTML = `
    <div class="phrase-speaker-dot"></div>
    <div class="phrase-content">
      <div class="phrase-meta">
        <span class="phrase-name">${escapeHtml(speakerName)}</span>
        <span class="lang-badge ${langClass}">${langLabel}</span>
        <span class="phrase-time">${phrase.timestamp || ""}</span>
      </div>
      <div class="phrase-text">${escapeHtml(phrase.text)}</div>
    </div>
  `;

  container.appendChild(el);
  scrollToBottom();
}

/**
 * Очищает область транскрипции и показывает пустое состояние.
 */
function clearTranscript() {
  const container = document.getElementById("transcript-scroll");
  // Удаляем все фразы но оставляем empty-state элемент
  container.querySelectorAll(".phrase").forEach(el => el.remove());
  const emptyState = document.getElementById("transcript-empty");
  if (emptyState) emptyState.style.display = "flex";
  appState.phraseCount = 0;
  updatePhraseCount(0);
}

/**
 * Прокручивает область транскрипции вниз.
 * Вызывается после каждой новой фразы.
 */
function scrollToBottom() {
  const container = document.getElementById("transcript-scroll");
  if (container) {
    container.scrollTop = container.scrollHeight;
  }
}

/**
 * Экранирует HTML-спецсимволы для безопасной вставки в innerHTML.
 * @param {string} str
 * @returns {string}
 */
function escapeHtml(str) {
  const el = document.createElement("div");
  el.textContent = str;
  return el.innerHTML;
}


// ─── Обновление UI ────────────────────────────────────────────────────────────

/**
 * Обновляет UI кнопки записи и индикатора статуса.
 * @param {boolean} isRecording
 */
function updateRecordingUI(isRecording) {
  const btn   = document.getElementById("record-btn");
  const dot   = document.getElementById("rec-dot");
  const label = document.getElementById("rec-label");

  if (!btn) return;

  if (isRecording) {
    btn.className = "btn btn-danger btn-lg";
    btn.setAttribute("data-i18n", "btn.stop");
    btn.textContent = t("btn.stop");
    dot.classList.add("active");
    label.setAttribute("data-i18n", "header.recording");
    label.textContent = t("header.recording");
    if (!appState._wasRecording) {
      showToast(t("toast.started"), "success");
      appState._wasRecording = true;
    }
  } else {
    btn.className = "btn btn-primary btn-lg";
    btn.setAttribute("data-i18n", "btn.start");
    btn.textContent = t("btn.start");
    dot.classList.remove("active");
    label.setAttribute("data-i18n", "header.idle");
    label.textContent = t("header.idle");
    if (appState._wasRecording) {
      showToast(t("toast.stopped"), "info");
      appState._wasRecording = false;
      loadSessionsList();  // Обновляем список сессий после остановки
    }
  }
}

/**
 * Обновляет счётчик фраз в нижней панели.
 * @param {number} count
 */
function updatePhraseCount(count) {
  const el = document.getElementById("phrase-count");
  if (el) el.textContent = `${count} ${t("phrases")}`;
}

/**
 * Обновляет индикатор статуса моделей в шапке.
 * @param {boolean} modelLoaded
 * @param {boolean} pyannoteLoaded
 */
function updateModelStatus(modelLoaded, pyannoteLoaded) {
  const el = document.getElementById("model-status");
  if (!el) return;

  if (!modelLoaded) {
    el.className = "model-status loading";
    el.textContent = t("models.loading");
    document.getElementById("record-btn")?.setAttribute("disabled", "true");
  } else {
    el.className = "model-status ready";
    el.textContent = pyannoteLoaded ? t("models.ready") : t("models.no_diarize");
    document.getElementById("record-btn")?.removeAttribute("disabled");
  }
}


// ─── Список сессий ────────────────────────────────────────────────────────────

/**
 * Загружает и отображает список сохранённых сессий в сайдбаре.
 */
async function loadSessionsList() {
  const container = document.getElementById("sessions-list");
  if (!container) return;

  try {
    const res = await fetch("/api/sessions");
    const sessions = await res.json();

    if (sessions.length === 0) {
      container.innerHTML = `<p class="sessions-empty" data-i18n="sessions.empty">${t("sessions.empty")}</p>`;
      return;
    }

    container.innerHTML = sessions.map(s => `
      <a class="session-item" href="/api/sessions/${encodeURIComponent(s.filename)}" target="_blank">
        <div class="session-item-date">${s.date}</div>
        <div class="session-item-size">${s.size_kb} KB</div>
      </a>
    `).join("");

  } catch (e) {
    container.innerHTML = `<p class="sessions-empty">—</p>`;
  }
}


// ─── Тосты ────────────────────────────────────────────────────────────────────

/**
 * Показывает всплывающее уведомление (тост).
 * Автоматически исчезает через duration мс.
 * @param {string} message
 * @param {"info"|"success"|"error"} type
 * @param {number} duration
 */
function showToast(message, type = "info", duration = 3000) {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.3s";
    setTimeout(() => toast.remove(), 300);
  }, duration);
}


// ─── Инициализация ────────────────────────────────────────────────────────────

/**
 * Инициализирует приложение при загрузке страницы.
 * Порядок: язык → переводы → имена спикеров → список сессий → WebSocket.
 */
function init() {
  // Язык
  currentLang = localStorage.getItem("ll_lang") || "ru";
  document.querySelectorAll(".ui-lang-switch button").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.lang === currentLang);
  });

  // Переводы
  applyTranslations();

  // Имена спикеров из localStorage
  loadSpeakerNames();

  // Список прошлых сессий
  loadSessionsList();

  // Статус моделей (до WebSocket подключения)
  updateModelStatus(false, false);

  // WebSocket
  initWebSocket();
}

// Запускаем при загрузке DOM
document.addEventListener("DOMContentLoaded", init);
