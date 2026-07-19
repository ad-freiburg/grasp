<script>
  import { onMount, onDestroy, tick } from 'svelte';
  import AppHeader from './AppHeader.svelte';
  import ConversationPane from './ConversationPane.svelte';
  import Composer from './Composer.svelte';
  import AppFooter from './AppFooter.svelte';
  import {
    TASKS,
    configEndpoint,
    kgEndpoint,
    wsEndpoint,
    saveSharedStateEndpoint,
    loadSharedStateEndpoint,
    sharePathForId
  } from '../constants.js';
  const VALID_TASK_IDS = new Set(TASKS.map((task) => task.id));

  // Read route parameters from the query string (set by nginx redirects).
  // e.g. ?share=abc123, ?kgs=wikidata+gptkb, ?task=sparql-qa
  function readQueryParams() {
    if (typeof window === 'undefined') return { loadId: null, kgs: [], task: null };
    const params = new URLSearchParams(window.location.search);
    // Check query param first (?share=abc123), then path (/share/abc123).
    const shareMatch = window.location.pathname.match(/\/share\/([^/]+)\/?$/);
    const shareId = params.get('share')?.trim() || (shareMatch ? decodeURIComponent(shareMatch[1]) : null);
    // KGs: check query param first, then path (if served from a KG URL).
    // The meta tag is injected by nginx to distinguish KG paths from the mount point.
    const rawKgs = params.get('kgs')?.trim() || '';
    let kgs = rawKgs
      ? rawKgs.split(',').map(s => decodeURIComponent(s.trim())).filter(Boolean)
      : [];
    if (kgs.length === 0 && document.querySelector('meta[name="grasp-kg-path"]')) {
      const lastSegment = window.location.pathname.replace(/\/+$/, '').split('/').pop();
      if (lastSegment) {
        kgs = lastSegment.split('+').map(s => decodeURIComponent(s.trim())).filter(Boolean);
      }
    }
    const taskParam = params.get('task')?.trim() || null;
    return {
      loadId: shareId,
      kgs,
      task: isValidTaskId(taskParam) ? taskParam : null
    };
  }

  const queryParams = readQueryParams();
  const initialKgSeed = sanitizeInitialKgs(queryParams.kgs);
  const initialTaskSeed = queryParams.task;
  const hasRouteProvidedSelections =
    initialKgSeed.length > 0 || Boolean(initialTaskSeed) || Boolean(queryParams.loadId);
  const shouldSkipStoredSelections = hasRouteProvidedSelections;

const STORAGE_KEYS = {
  task: 'grasp:task',
  selectedKgs: 'grasp:selectedKgs',
  lastOutput: 'grasp:lastOutput',
  lastInput: 'grasp:lastInput'
};
const SESSION_STORAGE_KEYS = {
  lastOutput: STORAGE_KEYS.lastOutput,
  lastInput: STORAGE_KEYS.lastInput
};

function getSessionStorage() {
  if (typeof window === 'undefined') return null;
  try {
    return window.sessionStorage;
  } catch (error) {
    console.warn('Session storage unavailable', error);
    return null;
  }
}

let composerValue = '';
let histories = [];
let task = initialTaskSeed || TASKS[0].id;
let knowledgeGraphs = new Map();
let sttEnabled = false;
let past = null;
let connectionStatus = 'initial';
let statusMessage = '';
let statusPinned = false;
let running = false;
  let cancelling = false;
  let socket;
  let persistedSelectedKgs = initialKgSeed;
  let composerWrapperEl;
  let composerOffset = 0;
  const COMPOSER_OFFSET_BUFFER = 0;
  let pendingCancelSignal = false;
  let pendingHistory = false;
  let pendingLoadId = queryParams.loadId;
  let lastInputRecord = null;
  let urlSelectedKgs = initialKgSeed.length ? [...initialKgSeed] : null;
  let urlSelectedTask = initialTaskSeed;
  let pendingUrlReset = false;
  let pendingTaskSwitch = null;

  $: pendingTaskSwitchName =
    TASKS.find((t) => t.id === pendingTaskSwitch)?.name ?? pendingTaskSwitch;

  $: hasHistory = histories.length > 0;
  $: knowledgeGraphList = Array.from(knowledgeGraphs.entries()).map(
    ([id, selected]) => ({ id, selected })
  );
  $: selectedKgs = knowledgeGraphList
    .filter((kg) => kg.selected)
    .map((kg) => kg.id);
  $: connected = connectionStatus === 'connected';
  $: disableControls =
    connectionStatus === 'initial' ||
    connectionStatus === 'connecting' ||
    connectionStatus === 'error' ||
    connectionStatus === 'disconnected';
  $: ceaInitialPayload = task === 'cea' ? getLastCeaInput() : null;
  $: elInitialPayload = task === 'entity-linking' ? getLastElInput() : null;
  const PAYLOAD_INPUT_TASKS = new Set(['cea', 'entity-linking']);
  onMount(async () => {
    await initialize();
    await measureComposerOnce();
  });

  onDestroy(() => {
    cleanupSocket();
  });

  async function initialize() {
    let sharedLoadAttempted = false;
    let sharedLoadSucceeded = false;
    if (pendingLoadId) {
      sharedLoadAttempted = true;
      sharedLoadSucceeded = await applySharedStateFromServer(pendingLoadId);
      pendingLoadId = null;
      scheduleUrlReset();
    }

    if (!sharedLoadAttempted || sharedLoadSucceeded) {
      restorePersistence();
    } else {
      resetStateForFailedShareLoad();
    }
    try {
      await Promise.all([loadKnowledgeGraphs(), loadServerConfig()]);
      await openConnection();
    } catch (error) {
      console.error('Failed to initialize', error);
      updateStatusMessage(
        formatStatusMessage(
          error,
          error?.status,
          'Failed to initialize. Please check your connection and reload.'
        )
      );
    }
  }

  function restorePersistence() {
    if (typeof window === 'undefined') return;
    try {
      if (!shouldSkipStoredSelections) {
        const storedTask = window.localStorage.getItem(STORAGE_KEYS.task);
        if (storedTask && TASKS.some((t) => t.id === storedTask)) {
          task = storedTask;
        }
        const storedKgs = window.localStorage.getItem(STORAGE_KEYS.selectedKgs);
        if (storedKgs) {
          persistedSelectedKgs = JSON.parse(storedKgs);
        }
      }
      const sessionStore = getSessionStorage();
      const storedOutput =
        sessionStore?.getItem(SESSION_STORAGE_KEYS.lastOutput) ?? null;
      if (storedOutput) {
        const parsed = JSON.parse(storedOutput);
        if (parsed) {
          past = {
            messages: parsed.pastMessages ?? [],
            known: parsed.pastKnown ?? []
          };
          if (Array.isArray(parsed.histories)) {
            histories = parsed.histories;
          }
        }
      }
      const storedInput =
        sessionStore?.getItem(SESSION_STORAGE_KEYS.lastInput) ?? null;
      if (storedInput && sessionStore) {
        const parsedInput = JSON.parse(storedInput);
        const isValidRecord =
          parsedInput &&
          typeof parsedInput === 'object' &&
          typeof parsedInput.task === 'string';
        if (isValidRecord && PAYLOAD_INPUT_TASKS.has(parsedInput.task)) {
          lastInputRecord = parsedInput;
        } else {
          sessionStore.removeItem(SESSION_STORAGE_KEYS.lastInput);
        }
      }
    } catch (error) {
      console.warn('Failed to restore persisted data', error);
    } finally {
      applyUrlStateOverrides();
      persistCurrentSelections();
    }
  }

  function applyUrlStateOverrides() {
    const hasTask = typeof urlSelectedTask === 'string' && urlSelectedTask;
    const hasKgs =
      Array.isArray(urlSelectedKgs) && urlSelectedKgs.length > 0;
    if (!hasTask && !hasKgs) {
      urlSelectedTask = null;
      urlSelectedKgs = null;
      return;
    }

    let shouldResetUrl = false;
    if (hasTask) {
      task = urlSelectedTask;
      persistTask(task);
      shouldResetUrl = true;
    }
    if (hasKgs) {
      persistedSelectedKgs = [...urlSelectedKgs];
      persistSelectedKgs(urlSelectedKgs);
      shouldResetUrl = true;
    }
    urlSelectedTask = null;
    urlSelectedKgs = null;
    if (shouldResetUrl) {
      scheduleUrlReset();
      clearHistory('full');
    }
  }

  function cloneCeaTable(table) {
    if (!table || typeof table !== 'object') return null;
    try {
      if (typeof structuredClone === 'function') {
        return structuredClone(table);
      }
    } catch (error) {
      console.warn('Failed to clone CEA table with structuredClone', error);
    }

    try {
      return JSON.parse(JSON.stringify(table));
    } catch (error) {
      console.warn('Failed to clone CEA table', error);
      return null;
    }
  }

  function cloneLastInputValue(value) {
    if (value == null) return null;
    if (typeof value !== 'object') return value;
    try {
      if (typeof structuredClone === 'function') {
        return structuredClone(value);
      }
    } catch {
      // fallthrough to JSON clone
    }
    try {
      return JSON.parse(JSON.stringify(value));
    } catch (error) {
      console.warn('Failed to clone last input value', error);
      return null;
    }
  }

  function getLastCeaInput() {
    if (!lastInputRecord || lastInputRecord.task !== 'cea') return null;
    if (
      lastInputRecord.value &&
      typeof lastInputRecord.value === 'object'
    ) {
      return cloneCeaTable(lastInputRecord.value) ?? lastInputRecord.value;
    }
    return null;
  }

  function getLastElInput() {
    if (!lastInputRecord || lastInputRecord.task !== 'entity-linking') {
      return null;
    }
    if (
      lastInputRecord.value &&
      typeof lastInputRecord.value === 'object' &&
      typeof lastInputRecord.value.data === 'string'
    ) {
      return cloneLastInputValue(lastInputRecord.value) ?? lastInputRecord.value;
    }
    return null;
  }

  async function loadServerConfig() {
    try {
      const response = await fetch(configEndpoint());
      if (!response.ok) return;
      const data = await response.json();
      sttEnabled = Boolean(data && data.speech_to_text);
    } catch (error) {
      console.warn('Failed to load server config', error);
    }
  }

  async function loadKnowledgeGraphs() {
    try {
      const response = await fetch(kgEndpoint());
      if (!response.ok) {
        throw createHttpError(response.status, 'Failed to load knowledge graphs.');
      }
      const available = await response.json();
      if (!Array.isArray(available) || available.length === 0) {
        throw new Error('No knowledge graphs available.');
      }

      const next = new Map();
      for (const kg of available) {
        const selected = persistedSelectedKgs.includes(kg);
        next.set(kg, selected);
      }

      if (![...next.values()].some(Boolean)) {
        if (next.has('wikidata')) {
          next.set('wikidata', true);
        } else {
          next.set(available[0], true);
        }
      }

      const selectedList = Array.from(next.entries())
        .filter(([, selected]) => selected)
        .map(([name]) => name);

      knowledgeGraphs = next;
      persistSelectedKgs(selectedList);
    } catch (error) {
      throw decorateError(error, 'Failed to load knowledge graphs.');
    }
  }

  async function openConnection() {
    cleanupSocket();
    connectionStatus = 'connecting';
    return new Promise((resolve, reject) => {
      try {
        socket = new WebSocket(wsEndpoint());
      } catch (error) {
        connectionStatus = 'error';
        return reject(decorateError(error, 'Failed to open WebSocket connection.'));
      }

      socket.addEventListener('open', () => {
        connectionStatus = 'connected';
        if (!statusPinned) {
          updateStatusMessage('');
        }
        resolve();
      });

      socket.addEventListener('message', handleSocketMessage);
      socket.addEventListener('close', handleSocketClose);
      socket.addEventListener('error', (event) => {
        connectionStatus = 'error';
        const decorated = decorateError(
          event?.error ?? new Error('WebSocket error occurred.'),
          'WebSocket error occurred.'
        );
        updateStatusMessage(
          formatStatusMessage(
            decorated,
            decorated.status,
            'WebSocket error occurred.'
          )
        );
        reject(decorated);
      });
    });
  }

  function cleanupSocket() {
    if (socket) {
      socket.removeEventListener('message', handleSocketMessage);
      socket.removeEventListener('close', handleSocketClose);
      socket.close();
      socket = null;
    }
  }

  function handleSocketClose(event) {
    connectionStatus = 'disconnected';
    running = false;
    cancelling = false;
    pendingCancelSignal = false;
    const reason =
      event?.reason ||
      'Connection to server lost. Please reload to reconnect.';
    updateStatusMessage(reason);
  }

  function handleSocketMessage(event) {
    try {
      const payload = JSON.parse(event.data);
      const hasType = Object.prototype.hasOwnProperty.call(payload, 'type');

      if (!hasType && payload.error) {
        updateStatusMessage(
          typeof payload.error === 'string'
            ? payload.error
            : 'Request failed.'
        );
        running = false;
        cancelling = false;
        pendingHistory = false;
        return;
      }

      if (!hasType && payload.cancelled) {
        clearHistory('last');
        pendingHistory = false;
        return;
      }

      if (!hasType) {
        return;
      }

      if (pendingHistory) {
        startNewHistory();
        pendingHistory = false;
      }

      if (payload.type === 'output') {
        maybePruneDuplicateReasoning(payload);
      }

      sendReceived();
      let enrichedPayload = payload;
      if (payload.type === 'output' && payload.task === 'cea') {
        const ceaInput = getLastCeaInput();
        if (ceaInput) {
          enrichedPayload = { ...payload, ceaInputTable: ceaInput };
        }
      }
      if (payload.type === 'output' && payload.task === 'entity-linking') {
        const elInput = getLastElInput();
        if (elInput) {
          enrichedPayload = { ...enrichedPayload, elInput };
        }
      }
      // the raw input message for entity linking contains the full prompt
      // scaffolding; attach the submitted payload so it can be rendered
      // as the plain text with the annotation window highlighted
      if (payload.type === 'input' && task === 'entity-linking') {
        const elInput = getLastElInput();
        if (elInput) {
          enrichedPayload = { ...enrichedPayload, elInput };
        }
      }
      if (payload.type === 'output') {
        enrichedPayload =
          enrichedPayload === payload
            ? { ...payload, shareLocked: false }
            : { ...enrichedPayload, shareLocked: false };
      }
      appendToCurrentHistory(enrichedPayload);

      if (enrichedPayload.type === 'output') {
        composerValue = '';
        cancelling = false;
        running = false;
        pendingCancelSignal = false;
        past = {
          messages: enrichedPayload.messages ?? [],
          known: enrichedPayload.known ?? []
        };
        persistLastOutput(enrichedPayload);
      }
    } catch (error) {
      console.error('Failed to handle message', error);
    }
  }

  function appendToCurrentHistory(item) {
    if (histories.length === 0) return;
    const lastIndex = histories.length - 1;
    histories = histories.map((history, index) =>
      index === lastIndex ? [...history, item] : history
    );
  }

  function maybePruneDuplicateReasoning(outputPayload) {
    if (outputPayload?.task !== 'general-qa') return;
    if (!outputPayload?.output) return;
    if (histories.length === 0) return;
    const lastIndex = histories.length - 1;
    const history = histories[lastIndex];
    if (!history || history.length === 0) return;
    const previous = history[history.length - 1];
    if (!previous || previous.type !== 'model') return;

    const reasoningText = (previous?.content ?? '').trim();
    const outputText =
      (outputPayload?.output?.output ?? outputPayload?.output?.answer ?? '')
        .trim();

    if (!reasoningText || !outputText) return;
    if (reasoningText !== outputText) return;

    histories = histories.map((historyItem, index) =>
      index === lastIndex ? historyItem.slice(0, -1) : historyItem
    );
  }

  function startNewHistory() {
    histories = [...histories, []];
  }


  function sendReceived() {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    const payload = { received: true };
    if (pendingCancelSignal) {
      payload.cancel = true;
      pendingCancelSignal = false;
    }
    socket.send(JSON.stringify(payload));
  }

  function handleSubmit(event) {
    if (running || !connected) return;
    if (!selectedKgs.length) return;

    let payloadInput = null;
    if (task === 'cea') {
      const detail = event.detail;
      if (!detail || detail.kind !== 'cea' || !detail.payload) return;
      payloadInput = detail.payload;
    } else if (task === 'entity-linking') {
      const detail = event.detail;
      if (!detail || detail.kind !== 'entity-linking' || !detail.payload) return;
      payloadInput = detail.payload;
    } else {
      const question = typeof event.detail === 'string' ? event.detail : '';
      const trimmedQuestion = question.trim();
      if (!trimmedQuestion) return;
      payloadInput = trimmedQuestion;
    }
    replaceUrlWithRoot();

    const clonedInput =
      task === 'cea'
        ? cloneCeaTable(payloadInput) ?? payloadInput
        : cloneLastInputValue(payloadInput);
    lastInputRecord = { task, value: clonedInput };
    persistLastInput(lastInputRecord);

    updateStatusMessage('');
    pendingHistory = true;
    running = true;
  const payload = {
      task,
      input: payloadInput,
      knowledge_graphs: selectedKgs,
      past: past ? { messages: past.messages, known: past.known } : null
    };
  try {
      socket?.send(JSON.stringify(payload));
      composerValue = '';
    } catch (error) {
      const decorated = decorateError(error, 'Failed to send request.');
      updateStatusMessage(
        formatStatusMessage(
          decorated,
          decorated.status,
          'Failed to send request.'
        )
      );
      running = false;
    }
  }

  function handleReset() {
    composerValue = '';
    updateStatusMessage('');
    clearHistory('full');
    replaceUrlWithRoot();
  }

  function handleCancel() {
    if (!connected) return;
    cancelling = true;
    pendingCancelSignal = true;
  }

  function handleTaskChange(event) {
    const nextTask = event.detail;
    if (!nextTask || task === nextTask) return;
    // switching tasks mid-conversation would send the follow-up with the
    // previous task's context (past messages and system prompt), so ask
    // for confirmation and clear the conversation first
    if (hasHistory || past) {
      pendingTaskSwitch = nextTask;
      return;
    }
    applyTaskChange(nextTask);
  }

  function applyTaskChange(nextTask) {
    task = nextTask;
    persistTask(task);
  }

  function confirmTaskSwitch() {
    const nextTask = pendingTaskSwitch;
    pendingTaskSwitch = null;
    if (!nextTask) return;
    composerValue = '';
    updateStatusMessage('');
    clearHistory('full');
    replaceUrlWithRoot();
    applyTaskChange(nextTask);
  }

  function cancelTaskSwitch() {
    pendingTaskSwitch = null;
  }

  function handleTaskSwitchKeydown(event) {
    if (pendingTaskSwitch === null) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      cancelTaskSwitch();
    }
  }

  function handleKnowledgeGraphChange(event) {
    const id = event.detail;
    if (!knowledgeGraphs.has(id)) return;
    const currentlySelected = knowledgeGraphs.get(id);
    if (
      currentlySelected &&
      selectedKgs.filter((kg) => kg !== id).length === 0
    ) {
      return;
    }

    const next = new Map(knowledgeGraphs);
    next.set(id, !currentlySelected);
    knowledgeGraphs = next;
    persistSelectedKgs(
      Array.from(next.entries())
        .filter(([, selected]) => selected)
        .map(([name]) => name)
    );
  }

  function clearHistory(mode) {
    cancelling = false;
    running = false;
    pendingCancelSignal = false;
    if (mode === 'full') {
      histories = [];
      past = null;
      clearLastOutput();
      lastInputRecord = null;
      clearLastInput();
    } else if (mode === 'last' && histories.length > 0) {
      histories = histories.slice(0, -1);
    }
  }

  function persistTask(value) {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(STORAGE_KEYS.task, value);
  }

  function persistSelectedKgs(values) {
    if (typeof window === 'undefined') return;
    window.localStorage.setItem(
      STORAGE_KEYS.selectedKgs,
      JSON.stringify(values)
    );
  }

  function persistLastOutput(outputMessage) {
    const sessionStore = getSessionStorage();
    if (!sessionStore) return;
    const payload = {
      pastMessages: outputMessage.messages ?? [],
      pastKnown: outputMessage.known ?? [],
      histories
    };
    sessionStore.setItem(
      SESSION_STORAGE_KEYS.lastOutput,
      JSON.stringify(payload)
    );
  }

  function persistLastInput(record) {
    const sessionStore = getSessionStorage();
    if (!sessionStore) return;
    if (!record || !PAYLOAD_INPUT_TASKS.has(record.task)) {
      sessionStore.removeItem(SESSION_STORAGE_KEYS.lastInput);
      return;
    }
    sessionStore.setItem(
      SESSION_STORAGE_KEYS.lastInput,
      JSON.stringify(record)
    );
  }

  function clearLastOutput() {
    const sessionStore = getSessionStorage();
    sessionStore?.removeItem(SESSION_STORAGE_KEYS.lastOutput);
  }

  function clearLastInput() {
    const sessionStore = getSessionStorage();
    sessionStore?.removeItem(SESSION_STORAGE_KEYS.lastInput);
  }

  function persistCurrentSelections() {
    if (typeof task === 'string' && task) {
      persistTask(task);
    }
    if (Array.isArray(persistedSelectedKgs)) {
      persistSelectedKgs(persistedSelectedKgs);
    }
  }

  function resetStateForFailedShareLoad() {
    histories = [];
    past = null;
    composerValue = '';
    lastInputRecord = null;
    persistedSelectedKgs = [];
  }

  function updateStatusMessage(message, options = {}) {
    const { pinned = false } = options;
    statusMessage = message;
    statusPinned = pinned;
  }

  function replaceUrlWithRoot() {
    scheduleUrlReset();
  }

  function scheduleUrlReset() {
    if (typeof window === 'undefined') return;
    if (pendingUrlReset) return;
    pendingUrlReset = true;
    setTimeout(async () => {
      pendingUrlReset = false;
      try {
        await tick();
        let cleanUrl = window.location.pathname;
        // Strip /share/... from the path.
        cleanUrl = cleanUrl.replace(/\/share\/.*$/, '/');
        // Strip KG path segment if served from a KG URL.
        const kgMeta = document.querySelector('meta[name="grasp-kg-path"]');
        if (kgMeta) {
          cleanUrl = cleanUrl.replace(/\/[^/]+\/?$/, '/');
          kgMeta.remove();
        }
        cleanUrl = cleanUrl.replace(/\/+$/, '') + '/';
        history.replaceState({}, '', cleanUrl);
        // Remove the <base> tag injected by nginx for /share/ URLs,
        // so subsequent fetch() calls resolve relative to the new URL.
        const baseTag = document.querySelector('base[href="../"]');
        if (baseTag) baseTag.remove();
      } catch (error) {
        console.warn('Failed to reset URL to root', error);
      }
    }, 0);
  }

  function reloadPage() {
    if (typeof window !== 'undefined') {
      window.location.reload();
    }
  }

  function buildSharePayload() {
    const selected = Array.isArray(selectedKgs)
      ? [...selectedKgs]
      : [];
    const snapshot = Array.isArray(histories)
      ? histories.map((items) =>
          items.map((entry) => ({ ...entry }))
        )
      : [];
    const shareInput =
      lastInputRecord &&
      PAYLOAD_INPUT_TASKS.has(lastInputRecord.task) &&
      lastInputRecord.task === task
        ? cloneLastInputValue(lastInputRecord.value)
        : null;
    return {
      task,
      selectedKgs: selected,
      lastInput: shareInput,
      lastOutput: {
        pastMessages: past?.messages ?? [],
        pastKnown: past?.known ?? [],
        histories: snapshot
      }
    };
  }

  async function createShareLink() {
    const payload = buildSharePayload();
    try {
      const response = await fetch(saveSharedStateEndpoint(), {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(payload)
      });
      if (!response.ok) {
        throw createHttpError(response.status, 'Failed to create share link.');
      }
      const result = await response.json();
      const id =
        typeof result?.id === 'string' && result.id.trim()
          ? result.id.trim()
          : '';
      const fallbackUrl =
        typeof result?.url === 'string' && result.url.trim()
          ? result.url.trim()
          : '';
      return {
        id,
        url: sharePathForId(id) || fallbackUrl
      };
    } catch (error) {
      const decorated = decorateError(error, 'Failed to create share link.');
      throw decorated;
    }
  }

  async function applySharedStateFromServer(id) {
    if (!id) return false;
    try {
      const response = await fetch(loadSharedStateEndpoint(id), {
        method: 'GET',
        headers: {
          'Accept': 'application/json'
        }
      });
      if (!response.ok) {
        throw createHttpError(
          response.status,
          'Failed to load shared conversation.'
        );
      }
      const payload = await response.json();
      persistSharedSnapshot(payload);
      return true;
    } catch (error) {
      const decorated = decorateError(
        error,
        'Failed to load shared conversation.'
      );
      const status = typeof decorated.status === 'number' ? decorated.status : undefined;
      const message =
        status === 404
          ? 'Shared conversation not found. The link might be invalid or the conversation has expired.'
          : formatStatusMessage(
              decorated,
              status,
              'Failed to load shared conversation.'
            );
      updateStatusMessage(message, { pinned: true });
      return false;
    }
  }

  function persistSharedSnapshot(payload) {
    if (typeof window === 'undefined' || !payload) return;
    const sessionStore = getSessionStorage();
    try {
      if (typeof payload.task === 'string' && isValidTaskId(payload.task)) {
        task = payload.task;
        persistTask(payload.task);
      }
      const sharedSelectedKgs = Array.isArray(payload.selectedKgs)
        ? payload.selectedKgs
        : Array.isArray(payload.selected_kgs)
          ? payload.selected_kgs
          : undefined;
      if (Array.isArray(sharedSelectedKgs)) {
        const sanitizedSharedKgs = sanitizeInitialKgs(sharedSelectedKgs);
        persistedSelectedKgs = sanitizedSharedKgs;
        persistSelectedKgs(sanitizedSharedKgs);
      }
      if (payload.lastOutput && typeof payload.lastOutput === 'object') {
        if (Array.isArray(payload.lastOutput.histories)) {
          payload.lastOutput.histories = markHistoriesAsShared(
            payload.lastOutput.histories
          );
        }
        sessionStore?.setItem(
          SESSION_STORAGE_KEYS.lastOutput,
          JSON.stringify(payload.lastOutput)
        );
      } else if (sessionStore) {
        sessionStore.removeItem(SESSION_STORAGE_KEYS.lastOutput);
      }
      const sharedLastInput = Object.prototype.hasOwnProperty.call(payload, 'lastInput')
        ? payload.lastInput
        : Object.prototype.hasOwnProperty.call(payload, 'last_input')
          ? payload.last_input
          : undefined;
      if (sharedLastInput !== undefined) {
        const targetTask =
          typeof payload.task === 'string' ? payload.task : task;
        if (sharedLastInput == null || !PAYLOAD_INPUT_TASKS.has(targetTask)) {
          sessionStore?.removeItem(SESSION_STORAGE_KEYS.lastInput);
          lastInputRecord = null;
        } else {
          const record = {
            task: targetTask,
            value: sharedLastInput
          };
          sessionStore?.setItem(
            SESSION_STORAGE_KEYS.lastInput,
            JSON.stringify(record)
          );
          lastInputRecord = record;
        }
      }
    } catch (error) {
      console.warn('Failed to persist shared snapshot', error);
    }
  }

  function markHistoriesAsShared(histories) {
    if (!Array.isArray(histories)) return [];
    return histories.map((history) => {
      if (!Array.isArray(history)) return history;
      return history.map((entry) => {
        if (entry && typeof entry === 'object' && entry.type === 'output') {
          return { ...entry, shareLocked: true };
        }
        return entry;
      });
    });
  }

  async function measureComposerOnce() {
    if (typeof window === 'undefined') return;
    await tick();
    composerOffset = COMPOSER_OFFSET_BUFFER;
  }

  function createHttpError(status, fallbackMessage) {
    const error = new Error(fallbackMessage);
    error.status = status;
    return error;
  }

  function decorateError(error, fallbackMessage) {
    const isError = error instanceof Error;
    const rawMessage = isError ? error.message : '';
    const initialStatus =
      isError && typeof error.status === 'number' ? error.status : undefined;
    const status = initialStatus ?? extractStatusCode(rawMessage);
    const message = formatStatusMessage(rawMessage, status, fallbackMessage);
    const decorated = new Error(message);
    if (status) {
      decorated.status = status;
    }
    return decorated;
  }

  function formatStatusMessage(rawMessage, status, fallbackMessage) {
    const rawString =
      typeof rawMessage === 'string'
        ? rawMessage
        : rawMessage && typeof rawMessage.message === 'string'
          ? rawMessage.message
          : '';
    const message = rawString.trim();
    const effectiveStatus = status ?? extractStatusCode(message);

    if (effectiveStatus && effectiveStatus >= 500 && effectiveStatus < 600) {
      return `Server error (${effectiveStatus}). Please try again in a moment.`;
    }

    if (effectiveStatus && effectiveStatus >= 400 && effectiveStatus < 500) {
      return `Request failed (${effectiveStatus}). Please check your input and try again.`;
    }

    if (!message || message.includes('Failed to fetch')) {
      return message
        ? 'Network error: Unable to reach the server. Please check your connection and try again.'
        : fallbackMessage || 'Unexpected error occurred.';
    }

    return message;
  }

  function extractStatusCode(message) {
    if (typeof message !== 'string') return undefined;
    const match = message.match(/\b([45]\d{2})\b/);
    if (!match) return undefined;
    const code = Number.parseInt(match[1], 10);
    return Number.isNaN(code) ? undefined : code;
  }

  function sanitizeInitialKgs(input) {
    if (!Array.isArray(input)) return [];
    const seen = new Set();
    const sanitized = [];
    for (const candidate of input) {
      if (typeof candidate !== 'string') continue;
      const trimmed = candidate.trim();
      if (!trimmed || seen.has(trimmed)) continue;
      seen.add(trimmed);
      sanitized.push(trimmed);
    }
    return sanitized;
  }

  function isValidTaskId(value) {
    return typeof value === 'string' && VALID_TASK_IDS.has(value);
  }

</script>

<svelte:window on:keydown={handleTaskSwitchKeydown} />

<section class="app-shell">
  <AppFooter />

  <div class="shell-content" class:shell-content--empty={!hasHistory}>
    <AppHeader />

    <main class="main-column" class:main-column--empty={!hasHistory} class:main-column--has-history={hasHistory}>
      {#if hasHistory}
        <ConversationPane
          {histories}
          {running}
          {cancelling}
          composerOffset={composerOffset}
          shareConversation={createShareLink}
          {selectedKgs}
        />
      {/if}

      <div class="composer-wrapper" class:composer-wrapper--sticky={hasHistory} bind:this={composerWrapperEl}>
        <Composer
          bind:value={composerValue}
          on:submit={handleSubmit}
          on:reset={handleReset}
          on:cancel={handleCancel}
          connected={connected}
          disabled={disableControls}
          isRunning={running}
          isCancelling={cancelling}
          task={task}
          tasks={TASKS}
          knowledgeGraphs={knowledgeGraphList}
          hasHistory={hasHistory}
          errorMessage={statusMessage}
          onReload={statusPinned ? null : reloadPage}
          on:taskchange={handleTaskChange}
          on:kgchange={handleKnowledgeGraphChange}
          initialCeaPayload={ceaInitialPayload}
          initialElPayload={elInitialPayload}
          {sttEnabled}
        />
      </div>
    </main>
  </div>
</section>

{#if pendingTaskSwitch !== null}
  <div
    class="task-switch-backdrop"
    role="presentation"
    on:pointerdown={cancelTaskSwitch}
  >
    <div
      class="task-switch-modal"
      role="dialog"
      aria-modal="true"
      aria-labelledby="task-switch-title"
      aria-describedby="task-switch-description"
      on:pointerdown|stopPropagation
      tabindex="-1"
    >
      <h2 class="task-switch-modal__title" id="task-switch-title">
        Switch to {pendingTaskSwitchName}?
      </h2>
      <p class="task-switch-modal__description" id="task-switch-description">
        Switching the task clears the current conversation and its context.
        Follow-up inputs will start from scratch.
      </p>
      <div class="task-switch-modal__actions">
        <button
          type="button"
          class="task-switch-modal__button task-switch-modal__button--secondary"
          on:click={cancelTaskSwitch}
        >
          Cancel
        </button>
        <button
          type="button"
          class="task-switch-modal__button task-switch-modal__button--primary"
          on:click={confirmTaskSwitch}
        >
          Switch and clear
        </button>
      </div>
    </div>
  </div>
{/if}

<style>
  .app-shell {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    padding: 12px 12px 0;
    margin: 0 auto;
    width: min(100%, 1040px);
    gap: var(--spacing-lg);
  }

  .shell-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: var(--spacing-lg);
  }

  .shell-content--empty {
    justify-content: center;
    align-items: center;
  }

  .main-column {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-lg);
    flex: 1;
  }

  .main-column--has-history {
    gap: var(--spacing-xs);
  }

  .main-column.main-column--empty {
    flex: 0 0 auto;
    display: flex;
    align-items: stretch;
    justify-content: center;
    flex-direction: column;
    gap: var(--spacing-lg);
    width: 100%;
  }

  .composer-wrapper {
    width: 100%;
  }

  .task-switch-backdrop {
    position: fixed;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: var(--spacing-lg);
    background: rgba(5, 17, 51, 0.45);
    z-index: 1000;
  }

  .task-switch-modal {
    background: #fff;
    border-radius: var(--radius-md);
    box-shadow: 0 20px 40px rgba(5, 17, 51, 0.2);
    max-width: 26rem;
    width: 100%;
    outline: none;
    display: grid;
    gap: var(--spacing-sm);
    padding: var(--spacing-xl);
  }

  .task-switch-modal__title {
    margin: 0;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--color-uni-blue);
  }

  .task-switch-modal__description {
    margin: 0;
    font-size: 0.9rem;
    color: var(--text-subtle);
  }

  .task-switch-modal__actions {
    display: flex;
    justify-content: flex-end;
    gap: var(--spacing-xs);
    margin-top: var(--spacing-xs);
  }

  .task-switch-modal__button {
    padding: 0.45rem 1rem;
    border-radius: var(--radius-sm);
    font-weight: 600;
    font: inherit;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
  }

  .task-switch-modal__button--secondary {
    border: 1px solid rgba(0, 0, 0, 0.15);
    background: #fff;
    color: var(--text-primary);
  }

  .task-switch-modal__button--primary {
    border: none;
    background: var(--color-uni-blue);
    color: #fff;
  }

  .task-switch-modal__button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 16px rgba(52, 74, 154, 0.15);
  }

  .composer-wrapper--sticky {
    position: sticky;
    bottom: 0;
    z-index: 10;
    padding-top: 0;
    background: linear-gradient(
      180deg,
      rgba(255, 255, 255, 0) 0%,
      rgba(255, 255, 255, 0.92) 55%,
      rgba(255, 255, 255, 1) 100%
    );
  }

</style>
