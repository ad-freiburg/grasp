<script>
  import { createEventDispatcher, onDestroy, onMount, tick } from 'svelte';
  import SelectionBar from './SelectionBar.svelte';
  import { parseCsvTable } from '../utils/csv.js';
  import { transcribeEndpoint } from '../constants.js';

  export let value = '';
  export let disabled = false;
  export let isRunning = false;
  export let isCancelling = false;
  export let connected = false;
  export let task = 'sparql-qa';
  export let tasks = [];
  export let knowledgeGraphs = [];
  export let hasHistory = false;
  export let errorMessage = '';
  export let onReload = null;
  export let initialCeaPayload = null;
  export let sttEnabled = false;
  const dispatch = createEventDispatcher();

  const MAX_FILE_SIZE_BYTES = 1024 * 1024;
  const MAX_COLUMNS = 100;
  const MAX_FILE_SIZE_LABEL = '1 MB';

  let textareaEl;
  let fileInputEl;
  let uploadButtonEl;
  let urlModalInputEl;
  let cachedLineHeight = 0;
  let isMobile = false;
  let previousValue = '';
  let isCeaTask = false;
  let ceaError = '';
  let ceaFileName = '';
  let ceaSummary = null;
  let ceaPayload = null;
  let isParsingFile = false;
  let lastTask = task;
  let ceaSelectedRows = [];
  let isUrlModalOpen = false;
  let urlModalInput = '';
  let urlModalError = '';
  let isUrlModalSubmitting = false;
  let ceaPreviousPayload = null;
  let ceaPreviousSummary = null;
  let ceaPreviousFileName = '';
  let ceaPreviousSelectedRows = [];
  let appliedInitialCeaRef = null;

  let isRecording = false;
  let isTranscribing = false;
  let sttError = '';
  let mediaRecorder = null;
  let recordingStream = null;
  let audioChunks = [];
  let recordingMimeType = '';

  const INACTIVITY_MESSAGE_PREFIX = 'connection closed due to inactivity';

  $: isCeaTask = task === 'cea';
  $: inputPlaceholder = task === 'sparql-to-question' ? 'Enter a SPARQL query...' : 'Ask a question...';
  $: trimmed = value.trim();
  $: canReload = typeof onReload === 'function';
  $: disableCeaInputs =
    disabled || isRunning || isCancelling || isParsingFile;
  $: disableFileInput = disableCeaInputs;
  $: disableRowSelection = disableCeaInputs;
  $: totalRowCount = ceaSummary?.rows ?? 0;
  $: selectedRowCount = ceaSelectedRows.length;
  $: annotateAllRows =
    totalRowCount > 0 && selectedRowCount === totalRowCount;
  $: annotateNone =
    totalRowCount > 0 ? selectedRowCount === 0 : false;
  $: selectedRowNumbers = ceaSelectedRows.map((index) => index + 1);
  $: selectedRowPreviewLabel =
    selectedRowNumbers.length > 0 && selectedRowNumbers.length <= 5
      ? selectedRowNumbers.join(', ')
      : selectedRowNumbers.length > 5
        ? `${selectedRowNumbers.length} rows`
        : '';
  $: canSubmit = isCeaTask
    ? Boolean(ceaPayload) &&
      selectedRowCount > 0 &&
      !disabled &&
      connected &&
      !isRunning &&
      !isCancelling &&
      !isParsingFile
      : trimmed.length > 0 &&
      !disabled &&
      connected &&
      !isRunning &&
      !isCancelling &&
      !isRecording &&
      !isTranscribing;
  $: canRecord = sttEnabled &&
    !isCeaTask &&
    !disabled &&
    !isRunning &&
    !isCancelling &&
    !isTranscribing;
  $: showMicControls = sttEnabled && !isCeaTask;
  $: canCancel = connected && isRunning && !isCancelling && !disabled;
  $: showCancel = isRunning || isCancelling;
  $: showClear = hasHistory && !isRunning && !isCancelling;
  $: normalizedErrorMessage =
    typeof errorMessage === 'string' ? errorMessage.trim() : '';
  $: inactivityDisconnect =
    normalizedErrorMessage.toLowerCase().startsWith(
      INACTIVITY_MESSAGE_PREFIX
    );
  $: hasError = Boolean(normalizedErrorMessage) && !inactivityDisconnect;
  $: showActions = !inactivityDisconnect;
  $: showReloadAction = inactivityDisconnect && canReload;
  $: cancelLabel = isCancelling ? 'Cancellation in progress' : 'Cancel';
  $: summaryRowsLabel = ceaSummary
    ? `${ceaSummary.rows} ${ceaSummary.rows === 1 ? 'row' : 'rows'}`
    : '';
  $: summaryColumnsLabel = ceaSummary
    ? `${ceaSummary.columns} ${ceaSummary.columns === 1 ? 'column' : 'columns'}`
    : '';
  $: hasPreviousCea = Boolean(ceaPreviousPayload) && Boolean(ceaPreviousSummary);

  $: if (isCeaTask) {
    if (initialCeaPayload && initialCeaPayload !== appliedInitialCeaRef) {
      applyInitialCea(initialCeaPayload);
      appliedInitialCeaRef = initialCeaPayload;
    }
  } else if (appliedInitialCeaRef) {
    appliedInitialCeaRef = null;
  }

  $: if (lastTask !== task) {
    if (lastTask === 'cea') {
      clearCeaSelection({ preservePrevious: true });
    }
    lastTask = task;
  }

  $: value, autoResize();
  $: if (!isCeaTask && textareaEl && value === '' && previousValue !== '') {
    focusInput();
  }
  $: previousValue = value;

  onMount(async () => {
    detectDevice();
    await tick();
    focusInput();
    if (!isCeaTask) {
      autoResize();
    }
  });

  function submit() {
    if (!canSubmit) return;
    if (isCeaTask) {
      const payload = buildCeaPayload();
      if (!payload) return;
      savePreviousCeaState();
      dispatch('submit', {
        kind: 'cea',
        payload,
        meta: {
          fileName: ceaFileName,
          rows: ceaSummary?.rows ?? 0,
          columns: ceaSummary?.columns ?? 0,
          selectedRows: selectedRowNumbers,
          selectionMode: annotateAllRows
            ? 'all'
            : annotateNone
              ? 'none'
              : 'partial'
        }
      });
      clearCeaSelection({ preservePrevious: true });
      return;
    }
    dispatch('submit', trimmed);
  }

  function cancel() {
    if (canCancel) {
      dispatch('cancel');
    }
  }

  function reset() {
    dispatch('reset');
    if (isCeaTask) {
      clearCeaSelection();
    }
    focusInput();
  }

  function onKeydown(event) {
    if (isCeaTask) {
      return;
    }
    if (event.key !== 'Enter') {
      return;
    }

    const ctrlOrMeta = event.ctrlKey || event.metaKey;

    if (isMobile) {
      if (ctrlOrMeta) {
        event.preventDefault();
        submit();
      }
      return;
    }

    if (event.shiftKey) {
      return;
    }

    event.preventDefault();
    submit();
  }

  function onTaskChange(event) {
    dispatch('taskchange', event.detail);
  }

  function onKgChange(event) {
    dispatch('kgchange', event.detail);
  }

  function autoResize() {
    if (!textareaEl) return;
    const style = getComputedStyle(textareaEl);
    if (!cachedLineHeight) {
      cachedLineHeight = parseFloat(style.lineHeight) || 20;
    }
    const padding =
      parseFloat(style.paddingTop || '0') + parseFloat(style.paddingBottom || '0');
    const minHeightFromStyle = parseFloat(style.minHeight || '0') || 0;
    const singleLineHeight = cachedLineHeight + padding;
    const minHeight = Math.max(singleLineHeight, minHeightFromStyle);
    const maxHeight = cachedLineHeight * 5 + padding;
    const trimmedValue = typeof value === 'string' ? value.trim() : '';
    textareaEl.style.height = 'auto';
    const contentHeight = textareaEl.scrollHeight;

    if (!trimmedValue) {
      textareaEl.style.height = `${minHeight}px`;
      textareaEl.style.overflowY = 'hidden';
      return;
    }

    const target = Math.min(Math.max(contentHeight, minHeight), maxHeight);
    textareaEl.style.height = `${target}px`;
    textareaEl.style.overflowY = contentHeight > maxHeight ? 'auto' : 'hidden';
  }

  function detectDevice() {
    if (typeof window === 'undefined') return;
    const coarse = window.matchMedia?.('(pointer: coarse)').matches;
    const nav = typeof navigator !== 'undefined' ? navigator : undefined;
    const uaData = nav?.userAgentData?.mobile;
    const uaString = nav?.userAgent ?? '';
    const uaFallback = /Mobi|Android|iP(ad|hone)/i.test(uaString);
    isMobile = Boolean(coarse || uaData || uaFallback);
  }

  function focusInput() {
    if (isCeaTask) {
      if (uploadButtonEl && !disableFileInput) {
        uploadButtonEl.focus();
      }
      return;
    }
    if (!textareaEl) return;
    textareaEl.focus();
  }

  function handleReload() {
    if (typeof onReload === 'function') {
      onReload();
    }
  }

  function pickRecordingMimeType() {
    if (typeof MediaRecorder === 'undefined') return '';
    const candidates = [
      'audio/webm;codecs=opus',
      'audio/webm',
      'audio/mp4',
      'audio/ogg;codecs=opus',
      'audio/ogg'
    ];
    for (const type of candidates) {
      if (MediaRecorder.isTypeSupported?.(type)) return type;
    }
    return '';
  }

  function releaseRecordingStream() {
    if (recordingStream) {
      for (const track of recordingStream.getTracks()) {
        track.stop();
      }
      recordingStream = null;
    }
    mediaRecorder = null;
    audioChunks = [];
  }

  async function startRecording() {
    if (!canRecord || isRecording) return;
    sttError = '';

    if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
      sttError = 'Microphone access is not supported in this browser.';
      return;
    }
    if (typeof MediaRecorder === 'undefined') {
      sttError = 'Audio recording is not supported in this browser.';
      return;
    }

    try {
      recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (error) {
      console.warn('Microphone access denied', error);
      sttError = 'Microphone access was denied.';
      return;
    }

    recordingMimeType = pickRecordingMimeType();
    try {
      mediaRecorder = recordingMimeType
        ? new MediaRecorder(recordingStream, { mimeType: recordingMimeType })
        : new MediaRecorder(recordingStream);
    } catch (error) {
      console.warn('Failed to start recorder', error);
      sttError = 'Failed to start recording.';
      releaseRecordingStream();
      return;
    }

    audioChunks = [];
    mediaRecorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        audioChunks.push(event.data);
      }
    };
    mediaRecorder.start();
    isRecording = true;
  }

  function cancelRecording() {
    if (!isRecording) return;
    try {
      mediaRecorder?.stop();
    } catch (error) {
      console.warn('Failed to stop recorder', error);
    }
    isRecording = false;
    releaseRecordingStream();
    focusInput();
  }

  async function stopAndTranscribe() {
    if (!isRecording || !mediaRecorder) return;
    const recorder = mediaRecorder;
    const mime = recordingMimeType || recorder.mimeType || 'audio/webm';

    const stopped = new Promise((resolve) => {
      recorder.addEventListener('stop', () => resolve(), { once: true });
    });

    try {
      recorder.stop();
    } catch (error) {
      console.warn('Failed to stop recorder', error);
      isRecording = false;
      releaseRecordingStream();
      sttError = 'Failed to stop recording.';
      return;
    }

    await stopped;
    isRecording = false;

    const chunks = audioChunks;
    releaseRecordingStream();

    if (!chunks.length) {
      sttError = 'No audio was captured.';
      return;
    }

    const blob = new Blob(chunks, { type: mime });
    if (!blob.size) {
      sttError = 'No audio was captured.';
      return;
    }

    isTranscribing = true;
    sttError = '';

    const extension = mime.includes('mp4')
      ? 'm4a'
      : mime.includes('ogg')
        ? 'ogg'
        : 'webm';

    const form = new FormData();
    form.append('file', blob, `recording.${extension}`);

    try {
      const response = await fetch(transcribeEndpoint(), {
        method: 'POST',
        body: form
      });
      if (!response.ok) {
        const message = response.status === 429
          ? 'Too many requests. Please try again later.'
          : 'Transcription failed.';
        throw new Error(message);
      }
      const data = await response.json();
      const text = typeof data?.text === 'string' ? data.text.trim() : '';
      if (text) {
        const current = typeof value === 'string' ? value : '';
        value = current && !/\s$/.test(current) ? `${current} ${text}` : `${current}${text}`;
        await tick();
        autoResize();
      }
    } catch (error) {
      console.warn('Transcription failed', error);
      sttError = error?.message || 'Transcription failed.';
    } finally {
      isTranscribing = false;
      focusInput();
    }
  }

  onDestroy(() => {
    if (isRecording) {
      try {
        mediaRecorder?.stop();
      } catch (error) {
        // ignore
      }
    }
    releaseRecordingStream();
  });

  function openFileDialog() {
    if (disableFileInput) return;
    fileInputEl?.click();
  }

  async function openUrlModal() {
    if (disableCeaInputs) return;
    isUrlModalOpen = true;
    urlModalInput = '';
    urlModalError = '';
    await tick();
    urlModalInputEl?.focus();
  }

  function closeUrlModal() {
    if (isUrlModalSubmitting || isParsingFile) return;
    isUrlModalOpen = false;
    urlModalInput = '';
    urlModalError = '';
    tick().then(() => {
      uploadButtonEl?.focus();
    });
  }

  function handleUrlModalBackdropClick() {
    closeUrlModal();
  }

  function handleUrlModalKeydown(event) {
    if (!isUrlModalOpen) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      closeUrlModal();
    }
  }

  function getByteSize(text) {
    if (typeof TextEncoder !== 'undefined') {
      return new TextEncoder().encode(text).length;
    }
    if (typeof Blob !== 'undefined') {
      return new Blob([text]).size;
    }
    return text.length;
  }

  function applyCsvContent({ text, sizeBytes }) {
    const byteLength =
      typeof sizeBytes === 'number' ? sizeBytes : getByteSize(text);

    if (byteLength > MAX_FILE_SIZE_BYTES) {
      throw new Error(
        `File is too large. Please choose a file smaller than ${MAX_FILE_SIZE_LABEL}.`
      );
    }

    const { header, rows } = parseCsvTable(text);
    const columnCount = header.length;

    if (columnCount > MAX_COLUMNS) {
      throw new Error(
        `This table has ${columnCount} columns. Please upload a table with at most ${MAX_COLUMNS} columns.`
      );
    }

    const data = rows.map((row) => row.slice());
    ceaPayload = { header, data };
    ceaSummary = { rows: data.length, columns: columnCount };
    ceaSelectedRows = [];
    ceaError = '';
  }

  function cloneCeaTable(table) {
    if (!table || typeof table !== 'object') return null;
    const header = Array.isArray(table.header) ? [...table.header] : [];
    const data = Array.isArray(table.data)
      ? table.data.map((row) => (Array.isArray(row) ? [...row] : []))
      : [];
    return { header, data };
  }

  function deriveCeaSelection(table, rowCount) {
    const rawAnnotate = Array.isArray(table?.annotate_rows)
      ? table.annotate_rows
      : Array.isArray(table?.annotateRows)
        ? table.annotateRows
        : null;
    const annotateRows = Array.isArray(rawAnnotate)
      ? rawAnnotate.filter((index) => Number.isInteger(index))
      : null;
    if (!rowCount) return [];
    if (annotateRows === null) {
      return Array.from({ length: rowCount }, (_, index) => index);
    }
    const maxIndex = rowCount - 1;
    return annotateRows
      .filter((index) => index >= 0 && index <= maxIndex)
      .sort((a, b) => a - b);
  }

  function applyInitialCea(table) {
    const cloned = cloneCeaTable(table);
    if (!cloned) return;
    const rows = Array.isArray(cloned.data) ? cloned.data : [];
    const header = Array.isArray(cloned.header) ? cloned.header : [];
    ceaPreviousPayload = cloned;
    ceaPreviousSummary = {
      rows: rows.length,
      columns: header.length
    };
    ceaPreviousSelectedRows = deriveCeaSelection(table, rows.length);
    const fileName =
      typeof table?.file_name === 'string'
        ? table.file_name
        : typeof table?.fileName === 'string'
          ? table.fileName
          : null;
    ceaPreviousFileName = fileName ?? 'Restored table';
    ceaPayload = null;
    ceaSummary = null;
    ceaSelectedRows = [];
    ceaFileName = '';
    ceaError = '';
  }

  function savePreviousCeaState() {
    if (!ceaPayload || !ceaSummary) return;
    ceaPreviousPayload = cloneCeaTable(ceaPayload);
    ceaPreviousSummary = { ...ceaSummary };
    ceaPreviousFileName = ceaFileName;
    ceaPreviousSelectedRows = [...ceaSelectedRows];
  }

  function clearCeaSelection(options = {}) {
    const { preservePrevious = false } = options;
    ceaPayload = null;
    ceaError = '';
    ceaFileName = '';
    ceaSummary = null;
    ceaSelectedRows = [];
    if (fileInputEl) {
      fileInputEl.value = '';
    }
    if (!preservePrevious) {
      ceaPreviousPayload = null;
      ceaPreviousSummary = null;
      ceaPreviousFileName = '';
      ceaPreviousSelectedRows = [];
    }
  }

  function restorePreviousCea() {
    if (!hasPreviousCea || disableCeaInputs) return;
    const table = cloneCeaTable(ceaPreviousPayload);
    if (!table) return;
    ceaPayload = table;
    ceaSummary = ceaPreviousSummary ? { ...ceaPreviousSummary } : null;
    ceaFileName = ceaPreviousFileName;
    ceaSelectedRows = Array.isArray(ceaPreviousSelectedRows)
      ? [...ceaPreviousSelectedRows]
      : [];
    ceaError = '';
  }

  async function handleFileChange(event) {
    const input = event.target;
    const [file] = input.files ?? [];
    ceaError = '';
    ceaSummary = null;
    ceaPayload = null;
    ceaSelectedRows = [];

    if (!file) {
      ceaFileName = '';
      return;
    }

    ceaFileName = file.name;

    if (file.size > MAX_FILE_SIZE_BYTES) {
      ceaError = `File is too large. Please choose a file smaller than ${MAX_FILE_SIZE_LABEL}.`;
      input.value = '';
      return;
    }

    if (
      file.type &&
      !file.type.includes('csv') &&
      !/\.csv$/i.test(file.name)
    ) {
      ceaError = 'Unsupported file type. Please provide a CSV file.';
      input.value = '';
      return;
    }

    isParsingFile = true;
    try {
      const text = await file.text();
      applyCsvContent({ text, sizeBytes: file.size });
    } catch (error) {
      ceaError = error?.message ?? 'Failed to read CSV file.';
      ceaPayload = null;
      ceaSummary = null;
      ceaSelectedRows = [];
    } finally {
      isParsingFile = false;
      input.value = '';
    }
  }

  async function importCsvFromUrl(url) {
    if (disableCeaInputs) {
      throw new Error('CSV input is currently disabled.');
    }

    let parsedUrl;
    try {
      parsedUrl = new URL(url);
    } catch (error) {
      const message = 'Please provide a valid URL.';
      ceaError = message;
      throw new Error(message);
    }

    const fileName =
      parsedUrl.pathname.split('/').filter(Boolean).pop() ||
      parsedUrl.hostname ||
      parsedUrl.toString();
    const urlString = parsedUrl.toString();

    isParsingFile = true;
    try {
      const response = await fetch(urlString);
      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`);
      }
      const text = await response.text();
      applyCsvContent({ text });
      ceaFileName = fileName;
    } catch (error) {
      const reason = error?.message?.trim();
      const message = reason
        ? reason.startsWith('Failed to load CSV from URL')
          ? reason
          : `Failed to load CSV from URL. ${reason}`
        : 'Failed to load CSV from URL.';
      ceaError = message;
      throw new Error(message);
    } finally {
      isParsingFile = false;
    }
  }

  async function submitUrlModal(event) {
    event?.preventDefault?.();
    if (isUrlModalSubmitting) return;
    const trimmedUrl = urlModalInput.trim();
    if (!trimmedUrl) {
      urlModalError = 'Please provide a URL.';
      urlModalInputEl?.focus();
      return;
    }

    urlModalError = '';
    isUrlModalSubmitting = true;
    try {
      await importCsvFromUrl(trimmedUrl);
      isUrlModalOpen = false;
      urlModalInput = '';
      await tick();
      uploadButtonEl?.focus();
    } catch (error) {
      const message = error?.message?.trim() || 'Failed to load CSV from URL.';
      urlModalError = message;
    } finally {
      isUrlModalSubmitting = false;
    }
  }

  async function importCsvFromClipboard() {
    if (disableCeaInputs) return;
    const nav = typeof navigator !== 'undefined' ? navigator : undefined;
    if (!nav?.clipboard?.readText) {
      ceaError = 'Clipboard access is not supported in this browser.';
      return;
    }

    ceaError = '';
    isParsingFile = true;
    try {
      const text = await nav.clipboard.readText();
      if (!text) {
        throw new Error('Clipboard does not contain any text.');
      }
      applyCsvContent({ text });
      ceaFileName = 'Clipboard';
    } catch (error) {
      const reason = error?.message?.trim();
      ceaError = reason
        ? reason.startsWith('Clipboard')
          ? reason
          : `Failed to read CSV from clipboard. ${reason}`
        : 'Failed to read CSV from clipboard.';
    } finally {
      isParsingFile = false;
    }
  }

  function buildCeaPayload() {
    if (!ceaPayload) return null;
    const annotateRows = annotateAllRows
      ? null
      : [...ceaSelectedRows].sort((a, b) => a - b);
    const payload = {
      header: ceaPayload.header,
      data: ceaPayload.data
    };
    if (annotateRows !== null) {
      payload.annotate_rows = annotateRows;
    }
    return payload;
  }

  function isRowSelected(index) {
    return ceaSelectedRows.includes(index);
  }

  function toggleRowSelection(index) {
    if (disableRowSelection || !ceaPayload) return;
    const next = ceaSelectedRows.includes(index)
      ? ceaSelectedRows.filter((value) => value !== index)
      : [...ceaSelectedRows, index];
    next.sort((a, b) => a - b);
    ceaSelectedRows = next;
  }

  function clearRowSelection() {
    if (disableRowSelection) return;
    ceaSelectedRows = [];
  }

  function selectAllRows() {
    if (disableRowSelection || !ceaPayload) return;
    ceaSelectedRows = ceaPayload.data.map((_, index) => index);
  }

  function handleRowKeydown(event, index) {
    if (disableRowSelection) return;
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      toggleRowSelection(index);
    }
  }
</script>

<svelte:window on:keydown={handleUrlModalKeydown} />

<form
  class="composer"
  class:composer--running={isRunning}
  on:submit|preventDefault={submit}
  aria-live="polite"
>
  {#if hasError}
    <div class="composer__alert" role="alert">
      <div class="composer__alert-text">
        <strong>Connection issue</strong>
        <span>{errorMessage}</span>
      </div>
      {#if canReload}
        <button
          type="button"
          class="composer__alert-button"
          on:click={handleReload}
        >
          Reload page
        </button>
      {/if}
    </div>
  {/if}
  <div class="composer__input-wrapper">
    <div class="composer__input-row">
      {#if isCeaTask}
        <div class="composer__upload-fieldset">
          <input
            class="composer__file-input"
            type="file"
            accept=".csv,text/csv"
            on:change={handleFileChange}
            bind:this={fileInputEl}
            disabled={disableFileInput}
          />
          <div class="composer__upload-controls">
            <div class="composer__upload-options">
              <button
                type="button"
                class="composer__upload-trigger"
                on:click={openFileDialog}
                disabled={disableFileInput}
                bind:this={uploadButtonEl}
              >
                {#if isParsingFile}
                  Reading CSV…
                {:else}
                  Upload file
                {/if}
              </button>
              <button
                type="button"
                class="composer__upload-trigger"
                on:click={openUrlModal}
                disabled={disableCeaInputs}
              >
                Load from URL
              </button>
              <button
                type="button"
                class="composer__upload-trigger"
                on:click={importCsvFromClipboard}
                disabled={disableCeaInputs}
              >
                Paste from Clipboard
              </button>
            </div>
            <span class="composer__upload-subtitle">
              CSV formatted tables up to 1MB and 100 columns are supported.
            </span>
          </div>
          {#if ceaPayload && ceaSummary}
            <p class="composer__file-info">
              <span class="composer__file-name">{ceaFileName}</span>
              <span class="composer__file-meta">
                {summaryRowsLabel} · {summaryColumnsLabel}
              </span>
            </p>
          {:else if ceaFileName}
            <p class="composer__file-info">
              <span class="composer__file-name">{ceaFileName}</span>
            </p>
          {/if}
          {#if ceaError}
            <p class="composer__error" role="alert">{ceaError}</p>
          {/if}
          {#if ceaPayload && ceaSummary}
            <div class="composer__preview" aria-live="polite">
              <div class="composer__preview-header">
                <div class="composer__preview-text">
                  <h3 class="composer__preview-title">CSV preview</h3>
                  <p class="composer__preview-status">
                    {#if annotateNone}
                      No rows selected. Click rows to include them in the annotation.
                    {:else if annotateAllRows}
                      All {totalRowCount} row{totalRowCount === 1 ? '' : 's'} selected. Click rows to exclude them.
                    {:else}
                      {selectedRowCount} row{selectedRowCount === 1 ? '' : 's'} selected
                      {#if selectedRowPreviewLabel}
                        ({selectedRowPreviewLabel})
                      {/if}
                      . Click a selected row to remove it.
                    {/if}
                  </p>
                </div>
                <div class="composer__preview-buttons">
                  <button
                    type="button"
                    class="composer__preview-button"
                    on:click={clearRowSelection}
                    disabled={disableRowSelection || annotateNone}
                  >
                    Clear selection
                  </button>
                  <button
                    type="button"
                    class="composer__preview-button"
                    on:click={selectAllRows}
                    disabled={disableRowSelection || selectedRowCount === totalRowCount}
                  >
                    Select all rows
                  </button>
                </div>
              </div>
              <div
                class="composer__preview-table"
                class:composer__preview-table--disabled={disableRowSelection}
                role="group"
                aria-label="CSV preview"
              >
                <table>
                  <thead>
                    <tr>
                      <th scope="col" class="composer__preview-index">Row</th>
                      {#each ceaPayload.header as column, columnIndex (columnIndex)}
                        <th scope="col">{column}</th>
                      {/each}
                    </tr>
                  </thead>
                  <tbody>
                    {#each ceaPayload.data as row, rowIndex (rowIndex)}
                      <tr
                        class:selected={isRowSelected(rowIndex)}
                        on:click={() => toggleRowSelection(rowIndex)}
                        on:keydown={(event) => handleRowKeydown(event, rowIndex)}
                        tabindex={disableRowSelection ? -1 : 0}
                        aria-selected={isRowSelected(rowIndex)}
                      >
                        <th scope="row" class="composer__preview-index">
                          {rowIndex + 1}
                        </th>
                        {#each row as cell, cellIndex (cellIndex)}
                          <td>{cell}</td>
                        {/each}
                      </tr>
                    {/each}
                  </tbody>
                </table>
              </div>
            </div>
          {:else if hasPreviousCea && !isRunning && !isCancelling}
            <div class="composer__reuse">
              <button
                type="button"
                class="composer__reuse-button"
                on:click={restorePreviousCea}
                disabled={disableCeaInputs}
              >
                Use previous table
              </button>
              {#if ceaPreviousFileName || (ceaPreviousSummary?.rows ?? 0)}
                <span class="composer__reuse-meta">
                  {#if ceaPreviousFileName}
                    {ceaPreviousFileName}
                  {/if}
                  {#if ceaPreviousSummary?.rows}
                    {' · '}
                    {ceaPreviousSummary.rows}
                    {ceaPreviousSummary.rows === 1 ? ' row' : ' rows'}
                  {/if}
                </span>
              {/if}
            </div>
          {/if}
        </div>
      {:else}
        <textarea
          id="composer-input"
          class="composer__input"
          placeholder={inputPlaceholder}
          bind:value
          bind:this={textareaEl}
          rows="1"
          on:keydown={onKeydown}
          on:input={autoResize}
        ></textarea>
      {/if}
      {#if showReloadAction}
        <div class="composer__input-actions">
          <button
            type="button"
            class="icon-button icon-button--danger icon-button--reload"
            on:click={handleReload}
            aria-label="Reconnect (connection closed due to inactivity)"
            title="Reconnect (connection closed due to inactivity)"
          >
            <span class="reload-icon" aria-hidden="true">↺</span>
          </button>
        </div>
      {:else if showActions}
        <div class="composer__input-actions">
      {#if showCancel}
            <button
              type="button"
              class="icon-button icon-button--danger"
              class:icon-button--cancelling={isCancelling}
              on:click={cancel}
              disabled={!canCancel}
              aria-label={cancelLabel}
              title={cancelLabel}
            >
              {#if isCancelling}
                <span class="cancel-spinner" aria-hidden="true"></span>
              {:else}
                <span class="cancel-icon" aria-hidden="true">✖</span>
              {/if}
            </button>
          {:else}
            {#if showMicControls}
              <button
                type="button"
                class="icon-button icon-button--mic"
                class:icon-button--mic-busy={isTranscribing}
                class:icon-button--mic-recording={isRecording}
                on:click={isRecording ? stopAndTranscribe : startRecording}
                disabled={!canRecord && !isRecording}
                aria-label={isRecording ? 'Stop and transcribe' : isTranscribing ? 'Transcribing…' : 'Record question'}
                title={sttError || (isRecording ? 'Stop and transcribe' : isTranscribing ? 'Transcribing…' : 'Record question')}
              >
                {#if isTranscribing}
                  <span class="transcribe-spinner" aria-hidden="true"></span>
                {:else}
                  <svg class="soundwave-icon" class:soundwave-icon--active={isRecording} viewBox="0 0 24 24" fill="none" aria-hidden="true">
                    <line class="soundwave-bar soundwave-bar--1" x1="4" y1="8" x2="4" y2="16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    <line class="soundwave-bar soundwave-bar--2" x1="8" y1="5" x2="8" y2="19" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    <line class="soundwave-bar soundwave-bar--3" x1="12" y1="2" x2="12" y2="22" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    <line class="soundwave-bar soundwave-bar--4" x1="16" y1="5" x2="16" y2="19" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                    <line class="soundwave-bar soundwave-bar--5" x1="20" y1="8" x2="20" y2="16" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                  </svg>
                {/if}
              </button>
            {/if}
            {#if isRecording}
              <button
                type="button"
                class="icon-button icon-button--danger"
                on:click={cancelRecording}
                aria-label="Discard recording"
                title="Discard recording"
              >
                <span class="cancel-icon" aria-hidden="true">✖</span>
              </button>
            {:else}
              <button
                type="button"
                class="icon-button icon-button--primary"
                on:click={submit}
                disabled={!canSubmit}
                aria-label="Run"
                title="Run"
              >
                <span class="paperplane-icon" aria-hidden="true">➤</span>
              </button>
            {/if}
          {/if}
          {#if showClear}
            <button
            type="button"
            class="icon-button icon-button--clear"
            on:click={reset}
            disabled={disabled}
            aria-label="Clear"
            title="Clear"
          >
              <span aria-hidden="true">↺</span>
            </button>
          {/if}
        </div>
      {/if}
    </div>
    {#if sttError}
      <p class="composer__error" role="alert">{sttError}</p>
    {/if}
  </div>

  <SelectionBar
    className="composer__selection"
    {task}
    {tasks}
    {knowledgeGraphs}
    compact={hasHistory}
    disabled={disabled || isRunning || isCancelling}
    on:taskchange={onTaskChange}
    on:kgchange={onKgChange}
  />
</form>

{#if isUrlModalOpen}
  <div
    class="composer__modal-backdrop"
    role="presentation"
    on:pointerdown={handleUrlModalBackdropClick}
  >
    <div
      class="composer__modal"
      role="dialog"
      aria-modal="true"
      aria-labelledby="composer-url-modal-title"
      on:pointerdown|stopPropagation
      tabindex="-1"
    >
      <form class="composer__modal-form" on:submit|preventDefault={submitUrlModal}>
        <h2 class="composer__modal-title" id="composer-url-modal-title">
          Load CSV from URL
        </h2>
        <p class="composer__modal-description">
          Paste the direct URL to a CSV file. The file must be publicly accessible.
        </p>
        <label class="composer__modal-label" for="composer-url-modal-input">
          CSV URL
        </label>
        <input
          id="composer-url-modal-input"
          class="composer__modal-input"
          type="url"
          name="csv-url"
          placeholder="https://example.com/data.csv"
          bind:value={urlModalInput}
          bind:this={urlModalInputEl}
          required
        />
        {#if urlModalError}
          <p class="composer__modal-error" role="alert">{urlModalError}</p>
        {/if}
        <div class="composer__modal-actions">
          <button
            type="button"
            class="composer__modal-button composer__modal-button--secondary"
            on:click={closeUrlModal}
            disabled={isUrlModalSubmitting || isParsingFile}
          >
            Cancel
          </button>
          <button
            type="submit"
            class="composer__modal-button composer__modal-button--primary"
            disabled={isUrlModalSubmitting || isParsingFile}
          >
            {#if isUrlModalSubmitting || isParsingFile}
              Loading…
            {:else}
              Load CSV
            {/if}
          </button>
        </div>
      </form>
    </div>
  </div>
{/if}

<style>
  .composer {
    display: grid;
    gap: var(--spacing-sm);
    background: var(--surface-base);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    padding: var(--spacing-lg);
    box-shadow: var(--shadow-sm);
    width: 100%;
    position: relative;
    overflow: hidden;
  }

  .composer::after {
    content: '';
    position: absolute;
    top: -1px;
    left: -1px;
    right: -1px;
    height: 3px;
    border-radius: var(--radius-md) var(--radius-md) 0 0;
    background: linear-gradient(
      90deg,
      rgba(52, 74, 154, 0) 0%,
      rgba(52, 74, 154, 0.9) 50%,
      rgba(52, 74, 154, 0) 100%
    );
    background-size: 200% 100%;
    opacity: 0;
    transition: opacity 0.2s ease;
    pointer-events: none;
  }

  .composer--running::after {
    opacity: 1;
    animation: composer-progress 1.2s linear infinite;
  }

  @keyframes composer-progress {
    from {
      background-position: 0% 0;
    }
    to {
      background-position: 200% 0;
    }
  }

  .composer__alert {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-sm);
    align-items: center;
    justify-content: space-between;
    border: 1px solid rgba(193, 0, 42, 0.25);
    background: rgba(193, 0, 42, 0.08);
    color: var(--color-uni-red);
    border-radius: var(--radius-sm);
    padding: var(--spacing-sm) var(--spacing-md);
  }

  .composer__alert-text {
    display: grid;
    gap: 2px;
  }

  .composer__alert-text strong {
    font-size: 0.95rem;
  }

  .composer__alert-text span {
    font-size: 0.85rem;
    color: var(--text-primary);
  }

  .composer__alert-button {
    padding: 0.4rem 1rem;
    border-radius: var(--radius-sm);
    border: none;
    background: var(--color-uni-blue);
    color: #fff;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    flex-shrink: 0;
  }

  .composer__alert-button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 14px rgba(52, 74, 154, 0.2);
  }

  .composer__alert-button:focus-visible {
    outline: 2px solid rgba(52, 74, 154, 0.4);
    outline-offset: 2px;
  }

  .composer__input-wrapper {
    display: flex;
    flex-direction: column;
  }

  .composer__input-row {
    display: flex;
    gap: var(--spacing-sm);
    align-items: stretch;
  }

  .composer__input {
    width: 100%;
    resize: none;
    min-height: 2.5rem;
    max-height: 10rem;
    border-radius: var(--radius-sm);
    border: 1px solid rgba(0, 0, 0, 0.12);
    padding: var(--spacing-sm) var(--spacing-md);
    font: inherit;
    line-height: 1.4;
    color: var(--text-primary);
    background: #fff;
    caret-color: var(--color-uni-blue);
  }

  .composer__input:focus {
    outline: none;
  }

  .composer__upload-fieldset {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: var(--spacing-xs);
    border: 1px dashed rgba(52, 74, 154, 0.35);
    border-radius: var(--radius-sm);
    padding: var(--spacing-md);
    background: rgba(52, 74, 154, 0.05);
  }

  .composer__upload-controls {
    display: grid;
    gap: var(--spacing-xs);
    align-items: flex-start;
  }

  .composer__upload-options {
    display: flex;
    align-items: center;
    gap: var(--spacing-xs);
    flex-wrap: wrap;
  }

  .composer__file-input {
    display: none;
  }

  .composer__upload-trigger {
    align-self: flex-start;
    padding: 0.5rem 1.1rem;
    border-radius: var(--radius-sm);
    border: 1px solid rgba(52, 74, 154, 0.28);
    background: var(--surface-base);
    color: var(--color-uni-blue);
    font: inherit;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
  }

  .composer__upload-trigger:not(:disabled):hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 12px rgba(52, 74, 154, 0.16);
  }

  .composer__upload-trigger:disabled {
    cursor: not-allowed;
    opacity: 0.6;
    transform: none;
    box-shadow: none;
  }

  .composer__upload-subtitle {
    margin: 0;
    font-size: 0.78rem;
    color: var(--text-subtle);
  }

  .composer__file-info {
    margin: 0;
    font-size: 0.85rem;
    color: var(--text-primary);
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
  }

  .composer__file-name {
    font-weight: 600;
  }

  .composer__file-meta {
    color: var(--text-subtle);
  }

  .composer__error {
    margin: 0;
    font-size: 0.85rem;
    color: var(--color-uni-red);
  }

  .composer__modal-backdrop {
    position: fixed;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: var(--spacing-lg);
    background: rgba(5, 17, 51, 0.45);
    z-index: 1000;
  }

  .composer__modal {
    background: #fff;
    border-radius: var(--radius-md);
    box-shadow: 0 20px 40px rgba(5, 17, 51, 0.2);
    max-width: 28rem;
    width: 100%;
    outline: none;
  }

  .composer__modal-form {
    display: grid;
    gap: var(--spacing-sm);
    padding: var(--spacing-xl);
  }

  .composer__modal-title {
    margin: 0;
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--color-uni-blue);
  }

  .composer__modal-description {
    margin: 0;
    font-size: 0.9rem;
    color: var(--text-subtle);
  }

  .composer__modal-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-primary);
  }

  .composer__modal-input {
    border: 1px solid rgba(0, 0, 0, 0.15);
    border-radius: var(--radius-sm);
    padding: 0.55rem 0.75rem;
    font: inherit;
    color: var(--text-primary);
  }

  .composer__modal-input:focus {
    outline: 2px solid rgba(52, 74, 154, 0.4);
    outline-offset: 2px;
  }

  .composer__modal-error {
    margin: 0;
    font-size: 0.85rem;
    color: var(--color-uni-red);
  }

  .composer__modal-actions {
    display: flex;
    justify-content: flex-end;
    gap: var(--spacing-xs);
    margin-top: var(--spacing-xs);
  }

  .composer__modal-button {
    padding: 0.45rem 1rem;
    border-radius: var(--radius-sm);
    font-weight: 600;
    font: inherit;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
  }

  .composer__modal-button--secondary {
    border: 1px solid rgba(0, 0, 0, 0.15);
    background: #fff;
    color: var(--text-primary);
  }

  .composer__modal-button--primary {
    border: none;
    background: var(--color-uni-blue);
    color: #fff;
  }

  .composer__modal-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }

  .composer__modal-button:not(:disabled):hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 16px rgba(52, 74, 154, 0.15);
  }

  .composer__preview {
    display: grid;
    gap: var(--spacing-sm);
    margin-top: var(--spacing-sm);
  }

  .composer__preview-header {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    gap: var(--spacing-sm);
    align-items: flex-start;
  }

  .composer__preview-text {
    display: grid;
    gap: 4px;
  }

  .composer__preview-title {
    margin: 0;
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--color-uni-blue);
  }

  .composer__preview-status {
    margin: 0;
    font-size: 0.85rem;
    color: var(--text-subtle);
  }

  .composer__preview-buttons {
    display: inline-flex;
    flex-wrap: wrap;
    gap: var(--spacing-xs);
  }

  .composer__preview-button {
    padding: 0.3rem 0.75rem;
    border-radius: var(--radius-sm);
    border: 1px solid rgba(52, 74, 154, 0.28);
    background: var(--surface-base);
    color: var(--color-uni-blue);
    font: inherit;
    font-size: 0.8rem;
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }

  .composer__preview-button:disabled {
    cursor: not-allowed;
    opacity: 0.5;
    transform: none;
    box-shadow: none;
  }

  .composer__preview-button:not(:disabled):hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 8px rgba(52, 74, 154, 0.14);
  }

  .composer__preview-table {
    border: 1px solid rgba(0, 0, 0, 0.12);
    border-radius: var(--radius-sm);
    overflow: hidden;
    background: #fff;
    max-height: 280px;
    overflow: auto;
  }

  .composer__preview-table table {
    width: 100%;
    border-collapse: collapse;
    min-width: 480px;
  }

  .composer__preview-table th,
  .composer__preview-table td {
    padding: 0.45rem 0.6rem;
    font-size: 0.85rem;
    text-align: left;
    border-bottom: 1px solid rgba(0, 0, 0, 0.06);
    vertical-align: top;
  }

  .composer__preview-table thead th {
    position: sticky;
    top: 0;
    z-index: 1;
    background: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(4px);
    font-weight: 600;
  }

  .composer__preview-index {
    width: 56px;
    white-space: nowrap;
  }

  .composer__preview-table tbody tr {
    cursor: pointer;
    transition: background 0.15s ease;
  }

  .composer__preview-table tbody tr:hover {
    background: rgba(52, 74, 154, 0.08);
  }

  .composer__preview-table tbody tr.selected {
    background: rgba(52, 74, 154, 0.18);
  }

  .composer__preview-table--disabled tbody tr {
    cursor: default;
  }

  .composer__preview-table--disabled tbody tr:hover {
    background: inherit;
  }

  .composer__input-actions {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-xs);
  }

  .icon-button {
    width: 2.1rem;
    height: 2.1rem;
    border-radius: var(--radius-sm);
    border: 1px solid transparent;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    padding: 0;
  }

  .icon-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }

  .icon-button--cancelling:disabled {
    opacity: 1;
    cursor: wait;
  }

  .icon-button--danger {
    background: rgba(193, 0, 42, 0.12);
    color: var(--color-uni-red);
    border: 1px solid rgba(193, 0, 42, 0.2);
    box-shadow: 0 4px 8px rgba(193, 0, 42, 0.1);
  }

  .icon-button--reload {
    width: 2.4rem;
    height: 2.4rem;
  }

  .icon-button--danger.icon-button--cancelling {
    background: rgba(193, 0, 42, 0.15);
    color: var(--color-uni-red);
    border: 1px solid rgba(193, 0, 42, 0.25);
    box-shadow: none;
  }

  .icon-button--danger.icon-button--cancelling .cancel-icon {
    display: none;
  }

  .icon-button--danger.icon-button--cancelling .cancel-spinner {
    display: inline-block;
  }

  .icon-button--primary {
    background: var(--color-uni-blue);
    color: #fff;
    box-shadow: 0 4px 8px rgba(52, 74, 154, 0.18);
  }

  .icon-button--primary:disabled {
    background: rgba(52, 74, 154, 0.35);
    color: rgba(255, 255, 255, 0.8);
    box-shadow: none;
  }

  .icon-button--clear {
    background: rgba(52, 74, 154, 0.12);
    color: var(--color-uni-blue);
    border: 1px solid rgba(52, 74, 154, 0.18);
    box-shadow: 0 4px 8px rgba(52, 74, 154, 0.16);
  }

  .icon-button--mic {
    background: rgba(52, 74, 154, 0.12);
    color: var(--color-uni-blue);
    border: 1px solid rgba(52, 74, 154, 0.18);
    box-shadow: 0 4px 8px rgba(52, 74, 154, 0.16);
  }

  .icon-button--mic-busy:disabled {
    opacity: 1;
    cursor: wait;
  }

  .icon-button--mic-recording {
    background: var(--color-uni-blue);
    color: #fff;
    border-color: transparent;
    box-shadow: 0 4px 8px rgba(52, 74, 154, 0.18);
  }

  .soundwave-icon {
    width: 1.15rem;
    height: 1.15rem;
    overflow: visible;
  }

  .soundwave-icon--active .soundwave-bar {
    transform-origin: center;
    animation: soundwave-bounce 0.8s ease-in-out infinite;
  }
  .soundwave-icon--active .soundwave-bar--1 { animation-delay: 0s; }
  .soundwave-icon--active .soundwave-bar--2 { animation-delay: 0.15s; }
  .soundwave-icon--active .soundwave-bar--3 { animation-delay: 0.3s; }
  .soundwave-icon--active .soundwave-bar--4 { animation-delay: 0.45s; }
  .soundwave-icon--active .soundwave-bar--5 { animation-delay: 0.6s; }

  @keyframes soundwave-bounce {
    0%, 100% { transform: scaleY(0.4); }
    50% { transform: scaleY(1); }
  }

  .icon-button:not(:disabled):hover {
    transform: translateY(-1px);
  }

  .reload-icon {
    font-size: 1.1rem;
    line-height: 1;
  }

  .cancel-spinner {
    width: 1.05rem;
    height: 1.05rem;
    border-radius: 50%;
    border: 2px solid rgba(193, 0, 42, 0.28);
    border-top-color: var(--color-uni-red);
    animation: spin 0.7s linear infinite;
  }

  .transcribe-spinner {
    width: 1.05rem;
    height: 1.05rem;
    border-radius: 50%;
    border: 2px solid rgba(52, 74, 154, 0.28);
    border-top-color: var(--color-uni-blue);
    animation: spin 0.7s linear infinite;
  }

  @keyframes spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }

  .paperplane-icon {
    font-size: 0.95rem;
    transform: translateY(-1px);
  }

  .composer__reuse {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--spacing-xs);
    background: rgba(52, 74, 154, 0.08);
    border: 1px solid rgba(52, 74, 154, 0.2);
    border-radius: var(--radius-sm);
    padding: var(--spacing-xs) var(--spacing-sm);
  }

  .composer__reuse-button {
    border: none;
    border-radius: var(--radius-sm);
    background: var(--color-uni-blue);
    color: #fff;
    font-weight: 600;
    padding: 0.35rem 0.85rem;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }

  .composer__reuse-button:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }

  .composer__reuse-button:not(:disabled):hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 12px rgba(52, 74, 154, 0.16);
  }

  .composer__reuse-button:focus-visible {
    outline: 2px solid rgba(52, 74, 154, 0.4);
    outline-offset: 2px;
  }

  .composer__reuse-meta {
    font-size: 0.85rem;
    color: var(--text-primary);
    font-weight: 500;
  }

  .composer__selection {
    margin-top: var(--spacing-sm);
  }

  @media (max-width: 600px) {
    .composer {
      padding: var(--spacing-md);
    }
  }

  .visually-hidden {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    border: 0;
  }
</style>
