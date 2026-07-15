<script>
  import { createEventDispatcher, onDestroy, onMount, tick } from 'svelte';
  import SelectionBar from './SelectionBar.svelte';
  import { parseCsvTable } from '../utils/csv.js';
  import { STT_TASKS, transcribeEndpoint } from '../constants.js';

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
  export let initialElPayload = null;
  export let sttEnabled = false;
  const dispatch = createEventDispatcher();

  const MAX_FILE_SIZE_BYTES = 1024 * 1024;
  const MAX_COLUMNS = 100;
  const MAX_FILE_SIZE_LABEL = '1 MB';
  const MAX_IMAGE_BYTES = 50 * 1048;
  const MAX_SELECTED_PDF_PAGES = 5;
  const PDF_RENDER_SCALE = 1.5;

  let textareaEl;
  let fileInputEl;
  let uploadButtonEl;
  let urlModalInputEl;
  let mediaInputEl;
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

  let isElTask = false;
  let elText = '';
  let elLastText = '';
  let elWindow = null;
  let elSpecialInstructions = '';
  let elTextareaEl;
  let elBackdropEl;
  let appliedInitialElRef = null;

  let isRecording = false;
  let isTranscribing = false;
  let sttError = '';
  let mediaRecorder = null;
  let recordingStream = null;
  let audioChunks = [];
  let recordingMimeType = '';
  let mediaError = '';
  let isConvertingPdf = false;
  let imageAttachments = [];
  let audioAttachments = [];
  let pdfPageAttachments = [];
  let mediaCounter = 0;
  let isDragOver = false;

  const INACTIVITY_MESSAGE_PREFIX = 'connection closed due to inactivity';

  $: isCeaTask = task === 'cea';
  $: isElTask = task === 'entity-linking';
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
    : isElTask
      ? normalizedElText.trim().length > 0 &&
        !disabled &&
        connected &&
        !isRunning &&
        !isCancelling &&
        !isRecording &&
        !isTranscribing
      : (trimmed.length > 0 || hasMediaAttachments) &&
      !pdfSelectionRequiredError &&
      !disabled &&
      connected &&
      !isRunning &&
      !isCancelling &&
      !isRecording &&
      !isTranscribing &&
      !isConvertingPdf;
  $: isSttTask = STT_TASKS.includes(task);
  $: canRecord = sttEnabled &&
    isSttTask &&
    !disabled &&
    !isRunning &&
    !isCancelling &&
    !isTranscribing;
  $: showMicControls = sttEnabled && isSttTask;
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
  $: selectedPdfPages = pdfPageAttachments.filter((page) => page.selected);
  $: selectedPdfPageCount = selectedPdfPages.length;
  $: selectedImagePayloads = [
    ...imageAttachments.map((item) => item.dataUrl),
    ...selectedPdfPages.map((item) => item.dataUrl)
  ];
  $: selectedAudioPayloads = audioAttachments.map((item) => item.dataUrl);
  $: hasMediaAttachments =
    selectedImagePayloads.length > 0 || selectedAudioPayloads.length > 0;
  $: pdfSelectionRequiredError =
    pdfPageAttachments.length > 0 && selectedPdfPageCount === 0
      ? `Select at least one PDF page (maximum ${MAX_SELECTED_PDF_PAGES}).`
      : '';

  $: if (isCeaTask) {
    if (initialCeaPayload && initialCeaPayload !== appliedInitialCeaRef) {
      applyInitialCea(initialCeaPayload);
      appliedInitialCeaRef = initialCeaPayload;
    }
  } else if (appliedInitialCeaRef) {
    appliedInitialCeaRef = null;
  }

  $: if (isElTask) {
    if (initialElPayload && initialElPayload !== appliedInitialElRef) {
      applyInitialEl(initialElPayload);
      appliedInitialElRef = initialElPayload;
    }
  } else if (appliedInitialElRef) {
    appliedInitialElRef = null;
  }

  $: if (lastTask !== task) {
    if (lastTask === 'cea') {
      clearCeaSelection({ preservePrevious: true });
    } else if (task === 'cea') {
      clearMediaAttachments();
    }
    lastTask = task;
  }

  // normalize like the backend does, so that the char offsets we compute on the
  // preview match the offsets the backend reports in its predictions
  $: normalizedElText = normalizeElText(elText);
  $: if (elText !== elLastText) {
    elLastText = elText;
    elWindow = null;
  }
  $: elWindowActive =
    isElTask &&
    elWindow !== null &&
    (elWindow.from > 0 || elWindow.to < elText.length);
  $: elWindowLabel = elWindowActive
    ? `Annotating only the highlighted part (characters ${elWindow.from}–${elWindow.to}), the rest is used as context.`
    : 'Annotating the whole text. Select a part of it above to only annotate that window.';

  $: value, autoResize();
  $: if (isElTask) {
    elText;
    elTextareaEl;
    elAutoResize();
  }
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
    if (isElTask) {
      const payload = buildElPayload();
      if (!payload) return;
      dispatch('submit', { kind: 'entity-linking', payload });
      // keep the text so that another window of it can be annotated next,
      // but clear the used window selection
      elWindow = null;
      return;
    }
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
    const multimodalPayload = {
      input: trimmed,
      image_input: selectedImagePayloads,
      audio_input: selectedAudioPayloads
    };
    dispatch('submit', multimodalPayload);
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
    } else {
      clearMediaAttachments();
    }
    if (isElTask) {
      clearElState();
    }
    focusInput();
  }

  function onKeydown(event) {
    if (isCeaTask || isElTask) {
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

  function resizeTextarea(el, content, maxLines = 5) {
    if (!el) return;
    const style = getComputedStyle(el);
    const lineHeight = parseFloat(style.lineHeight) || 20;
    const padding =
      parseFloat(style.paddingTop || '0') + parseFloat(style.paddingBottom || '0');
    const minHeightFromStyle = parseFloat(style.minHeight || '0') || 0;
    const singleLineHeight = lineHeight + padding;
    const minHeight = Math.max(singleLineHeight, minHeightFromStyle);
    const maxHeight = lineHeight * maxLines + padding;
    const trimmedContent = typeof content === 'string' ? content.trim() : '';
    el.style.height = 'auto';
    const contentHeight = el.scrollHeight;

    if (!trimmedContent) {
      el.style.height = `${minHeight}px`;
      el.style.overflowY = 'hidden';
      return;
    }

    const target = Math.min(Math.max(contentHeight, minHeight), maxHeight);
    el.style.height = `${target}px`;
    el.style.overflowY = contentHeight > maxHeight ? 'auto' : 'hidden';
  }

  function autoResize() {
    resizeTextarea(textareaEl, value);
  }

  function elAutoResize() {
    resizeTextarea(elTextareaEl, elText);
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
    if (isElTask) {
      elTextareaEl?.focus();
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
        if (isElTask) {
          const current = typeof elText === 'string' ? elText : '';
          elText = current && !/\s$/.test(current)
            ? `${current} ${text}`
            : `${current}${text}`;
          await tick();
          elAutoResize();
        } else {
          const current = typeof value === 'string' ? value : '';
          value = current && !/\s$/.test(current) ? `${current} ${text}` : `${current}${text}`;
          await tick();
          autoResize();
        }
      }
    } catch (error) {
      console.warn('Transcription failed', error);
      sttError = error?.message || 'Transcription failed.';
    } finally {
      isTranscribing = false;
      focusInput();
    }
  }

  function createMediaId(prefix) {
    mediaCounter += 1;
    return `${prefix}-${Date.now()}-${mediaCounter}`;
  }

  function clearMediaInput(input) {
    if (input) {
      input.value = '';
    }
  }

  function clearMediaError() {
    mediaError = '';
  }

  function clearMediaAttachments() {
    imageAttachments = [];
    audioAttachments = [];
    pdfPageAttachments = [];
    isConvertingPdf = false;
    isDragOver = false;
    mediaError = '';
    clearMediaInput(mediaInputEl);
  }

  function openMediaDialog() {
    if (isCeaTask || disabled || isRunning || isCancelling || isConvertingPdf) return;
    mediaInputEl?.click();
  }

  function removeImageAttachment(id) {
    imageAttachments = imageAttachments.filter((item) => item.id !== id);
  }

  function removeAudioAttachment(id) {
    audioAttachments = audioAttachments.filter((item) => item.id !== id);
  }

  function clearPdfAttachments() {
    pdfPageAttachments = [];
    clearMediaInput(mediaInputEl);
  }

  function removePdfPageAttachment(id) {
    pdfPageAttachments = pdfPageAttachments.filter((item) => item.id !== id);
  }

  function togglePdfPageSelection(id) {
    const target = pdfPageAttachments.find((page) => page.id === id);
    if (!target) return;
    if (!target.selected && selectedPdfPageCount >= MAX_SELECTED_PDF_PAGES) {
      mediaError = `You can select up to ${MAX_SELECTED_PDF_PAGES} PDF pages.`;
      return;
    }
    clearMediaError();
    pdfPageAttachments = pdfPageAttachments.map((page) =>
      page.id === id ? { ...page, selected: !page.selected } : page
    );
  }

  function fileToDataUrl(file) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error(`Failed to read file ${file.name}.`));
      reader.onload = () => {
        const result = typeof reader.result === 'string' ? reader.result : '';
        if (!result) {
          reject(new Error(`Failed to read file ${file.name}.`));
          return;
        }
        resolve(result);
      };
      reader.readAsDataURL(file);
    });
  }

  function getDataUrlByteSize(dataUrl) {
    const parts = dataUrl.split(',', 2);
    if (parts.length < 2) return Number.POSITIVE_INFINITY;
    const payload = parts[1];
    const padding = payload.endsWith('==') ? 2 : payload.endsWith('=') ? 1 : 0;
    return Math.floor((payload.length * 3) / 4) - padding;
  }

  async function canvasToJpegDataUrl(canvas, quality) {
    return new Promise((resolve, reject) => {
      canvas.toBlob(
        async (blob) => {
          if (!blob) {
            reject(new Error('Failed to render PDF page to image.'));
            return;
          }
          const dataUrl = await fileToDataUrl(blob);
          resolve(dataUrl);
        },
        'image/jpeg',
        quality
      );
    });
  }

  async function renderPdfPageToJpeg(page) {
    let scale = PDF_RENDER_SCALE;
    let quality = 0.82;

    for (let attempt = 0; attempt < 8; attempt += 1) {
      const viewport = page.getViewport({ scale });
      const canvas = document.createElement('canvas');
      const context = canvas.getContext('2d', { alpha: false });
      if (!context) {
        throw new Error('Failed to create PDF rendering context.');
      }
      canvas.width = Math.max(1, Math.floor(viewport.width));
      canvas.height = Math.max(1, Math.floor(viewport.height));

      await page.render({ canvasContext: context, viewport }).promise;
      const dataUrl = await canvasToJpegDataUrl(canvas, quality);
      const byteSize = getDataUrlByteSize(dataUrl);
      if (byteSize <= MAX_IMAGE_BYTES) {
        return {
          dataUrl,
          byteSize,
          width: canvas.width,
          height: canvas.height
        };
      }

      quality = Math.max(0.35, quality - 0.1);
      scale = Math.max(0.4, scale * 0.82);
    }

    throw new Error(
      `Unable to reduce PDF page below ${MAX_IMAGE_BYTES} bytes. Try a simpler document.`
    );
  }

  async function loadPdfModule() {
    const pdfjs = await import('pdfjs-dist/build/pdf.mjs');
    const worker = await import('pdfjs-dist/build/pdf.worker.min.mjs?url');
    pdfjs.GlobalWorkerOptions.workerSrc = worker.default;
    return pdfjs;
  }

  function classifyMediaFile(file) {
    const fileType = typeof file?.type === 'string' ? file.type.toLowerCase() : '';
    const fileName = typeof file?.name === 'string' ? file.name.toLowerCase() : '';
    if (fileType.startsWith('image/')) return 'image';
    if (fileType === 'application/pdf' || /\.pdf$/i.test(fileName)) return 'pdf';
    if (fileType.startsWith('audio/')) return 'audio';
    if (/\.(mp3|wav|ogg|webm|m4a|flac)$/i.test(fileName)) return 'audio';
    return 'unsupported';
  }

  async function appendImageFiles(files) {
    const next = [];
    const warnings = [];
    for (const file of files) {
      try {
        const dataUrl = await fileToDataUrl(file);
        if (getDataUrlByteSize(dataUrl) > MAX_IMAGE_BYTES) {
          throw new Error(
            `${file.name} exceeds the ${Math.floor(MAX_IMAGE_BYTES / 1024)}KB image limit.`
          );
        }
        next.push({
          id: createMediaId('image'),
          name: file.name,
          type: file.type,
          dataUrl
        });
      } catch (error) {
        warnings.push(error?.message ?? `Failed to load ${file.name}.`);
      }
    }
    if (next.length > 0) {
      imageAttachments = [...imageAttachments, ...next];
    }
    return warnings;
  }

  async function appendAudioFiles(files) {
    const next = [];
    const warnings = [];
    for (const file of files) {
      try {
        const dataUrl = await fileToDataUrl(file);
        next.push({
          id: createMediaId('audio'),
          name: file.name,
          type: file.type || 'audio/*',
          dataUrl
        });
      } catch (error) {
        warnings.push(error?.message ?? `Failed to load ${file.name}.`);
      }
    }
    if (next.length > 0) {
      audioAttachments = [...audioAttachments, ...next];
    }
    return warnings;
  }

  async function appendPdfFiles(files) {
    if (!files.length) return [];
    const warnings = [];
    const pages = [];
    let selectedCount = selectedPdfPageCount;

    isConvertingPdf = true;
    try {
      const pdfjs = await loadPdfModule();
      for (const file of files) {
        try {
          const data = await file.arrayBuffer();
          const loadingTask = pdfjs.getDocument({ data });
          const pdf = await loadingTask.promise;
          for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
            const page = await pdf.getPage(pageNumber);
            const rendered = await renderPdfPageToJpeg(page);
            const canSelect = selectedCount < MAX_SELECTED_PDF_PAGES;
            pages.push({
              id: createMediaId('pdf-page'),
              fileName: file.name,
              name: `${file.name} page ${pageNumber}`,
              pageNumber,
              selected: canSelect,
              ...rendered
            });
            if (canSelect) {
              selectedCount += 1;
            }
          }
        } catch (error) {
          warnings.push(error?.message ?? `Failed to convert ${file.name}.`);
        }
      }
      if (pages.length > 0) {
        pdfPageAttachments = [...pdfPageAttachments, ...pages];
      }
    } finally {
      isConvertingPdf = false;
    }

    return warnings;
  }

  async function processMediaFiles(fileLikeList) {
    const files = Array.from(fileLikeList ?? []);
    if (!files.length) return;
    clearMediaError();

    const images = [];
    const audio = [];
    const pdfs = [];
    const unsupported = [];

    for (const file of files) {
      const kind = classifyMediaFile(file);
      if (kind === 'image') {
        images.push(file);
      } else if (kind === 'audio') {
        audio.push(file);
      } else if (kind === 'pdf') {
        pdfs.push(file);
      } else {
        unsupported.push(file.name || 'unnamed file');
      }
    }

    const warnings = [];
    if (unsupported.length > 0) {
      warnings.push(
        `Unsupported media type discarded: ${unsupported.join(', ')}.`
      );
    }

    warnings.push(...(await appendImageFiles(images)));
    warnings.push(...(await appendAudioFiles(audio)));
    warnings.push(...(await appendPdfFiles(pdfs)));

    mediaError = warnings.filter(Boolean).join(' ');
  }

  async function handleMediaInputChange(event) {
    const files = Array.from(event.target.files ?? []);
    clearMediaInput(event.target);
    if (!files.length) return;
    await processMediaFiles(files);
  }

  function isMediaInputBlocked() {
    return isCeaTask || disabled || isRunning || isCancelling || isConvertingPdf;
  }

  function handleMediaDragEnter(event) {
    if (isCeaTask) return;
    event.preventDefault();
    if (isMediaInputBlocked()) return;
    isDragOver = true;
  }

  function handleMediaDragOver(event) {
    if (isCeaTask) return;
    event.preventDefault();
    if (isMediaInputBlocked()) return;
    if (event.dataTransfer) {
      event.dataTransfer.dropEffect = 'copy';
    }
  }

  function handleMediaDragLeave(event) {
    if (isCeaTask) return;
    event.preventDefault();
    const nextTarget = event.relatedTarget;
    if (nextTarget && event.currentTarget?.contains?.(nextTarget)) {
      return;
    }
    isDragOver = false;
  }

  async function handleMediaDrop(event) {
    if (isCeaTask) return;
    event.preventDefault();
    isDragOver = false;
    if (isMediaInputBlocked()) return;
    const files = Array.from(event.dataTransfer?.files ?? []);
    if (!files.length) return;
    await processMediaFiles(files);
  }

  async function handleMediaPaste(event) {
    if (isMediaInputBlocked()) return;
    const clipboardData = event.clipboardData;
    if (!clipboardData) return;
    const fromItems = Array.from(clipboardData.items ?? [])
      .filter((item) => item.kind === 'file')
      .map((item) => item.getAsFile())
      .filter(Boolean);
    const files = fromItems.length > 0 ? fromItems : Array.from(clipboardData.files ?? []);
    if (!files.length) return;
    event.preventDefault();
    await processMediaFiles(files);
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

  function normalizeElText(value) {
    if (typeof value !== 'string') return '';
    let normalized = value;
    try {
      normalized = normalized.normalize('NFC');
    } catch (error) {
      console.warn('Failed to NFC-normalize text', error);
    }
    return normalized.replace(/[‘’]/g, "'");
  }

  function applyInitialEl(payload) {
    if (!payload || typeof payload !== 'object') return;
    if (typeof payload.data !== 'string' || !payload.data) return;
    elText = payload.data;
    // keep elLastText in sync so the reactive window reset does not fire
    elLastText = elText;
    // restored data was normalized on submit, so raw and normalized
    // offsets are identical here
    const length = elText.length;
    const from = payload.annotate_from;
    const to = payload.annotate_up_to;
    if (
      Number.isInteger(from) &&
      Number.isInteger(to) &&
      from >= 0 &&
      to > from &&
      to <= length
    ) {
      elWindow = { from, to };
    } else {
      elWindow = null;
    }
    elSpecialInstructions =
      typeof payload.special_instructions === 'string'
        ? payload.special_instructions
        : '';
  }

  function clearElState() {
    elText = '';
    elLastText = '';
    elWindow = null;
    elSpecialInstructions = '';
  }

  function handleElSelect() {
    if (!elTextareaEl || disabled || isRunning || isCancelling) return;
    const start = elTextareaEl.selectionStart;
    const end = elTextareaEl.selectionEnd;
    if (!Number.isInteger(start) || !Number.isInteger(end)) return;
    if (end <= start) return;
    elWindow = { from: start, to: end };
  }

  function clearElWindow() {
    elWindow = null;
  }

  function syncElScroll() {
    if (elBackdropEl && elTextareaEl) {
      elBackdropEl.scrollTop = elTextareaEl.scrollTop;
    }
  }

  function buildElPayload() {
    const data = normalizedElText;
    if (!data.trim()) return null;
    const payload = { data };
    if (elWindowActive) {
      // elWindow offsets refer to the raw text; convert them to offsets into
      // the normalized text that is actually submitted
      const from = normalizeElText(elText.slice(0, elWindow.from)).length;
      const to =
        from + normalizeElText(elText.slice(elWindow.from, elWindow.to)).length;
      if (from < data.length && to > from) {
        payload.annotate_from = from;
        payload.annotate_up_to = Math.min(to, data.length);
      }
    }
    const instructions = elSpecialInstructions.trim();
    if (instructions) {
      payload.special_instructions = instructions;
    }
    return payload;
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
      {:else if isElTask}
        <div class="composer__el-fieldset">
          <div class="composer__el-input-wrap">
            <div
              class="composer__el-backdrop"
              bind:this={elBackdropEl}
              aria-hidden="true"
            >{#if elWindowActive}{elText.slice(0, elWindow.from)}<mark class="composer__el-window">{elText.slice(elWindow.from, elWindow.to)}</mark>{elText.slice(elWindow.to)}{:else}{elText}{/if}{'\n'}</div>
            <textarea
              id="composer-el-input"
              class="composer__el-input"
              placeholder="Enter or paste the text to annotate..."
              bind:value={elText}
              bind:this={elTextareaEl}
              rows="1"
              on:input={elAutoResize}
              on:select={handleElSelect}
              on:scroll={syncElScroll}
              disabled={disabled || isRunning || isCancelling}
            ></textarea>
          </div>
          {#if elText.trim()}
            <div class="composer__el-window-bar">
              <p class="composer__el-window-status">{elWindowLabel}</p>
              {#if elWindowActive}
                <button
                  type="button"
                  class="composer__preview-button"
                  on:click={clearElWindow}
                  disabled={disabled || isRunning || isCancelling}
                >
                  Annotate whole text
                </button>
              {/if}
            </div>
            <label class="composer__el-instructions">
              <span class="composer__el-instructions-label">
                Special instructions (optional)
              </span>
              <input
                type="text"
                class="composer__el-instructions-input"
                placeholder="e.g. only annotate persons and locations"
                bind:value={elSpecialInstructions}
                disabled={disabled || isRunning || isCancelling}
              />
            </label>
          {/if}
        </div>
      {:else}
        <div class="composer__multimodal-fieldset">
          <div
            class="composer__multimodal-input-row"
            class:composer__multimodal-input-row--drag-over={isDragOver}
            role="group"
            aria-label="Message input with media attachments"
            on:dragenter={handleMediaDragEnter}
            on:dragover={handleMediaDragOver}
            on:dragleave={handleMediaDragLeave}
            on:drop={handleMediaDrop}
          >
            <button
              type="button"
              class="composer__media-plus"
              on:click={openMediaDialog}
              disabled={disabled || isRunning || isCancelling || isConvertingPdf}
              aria-label="Attach media"
              title="Attach media"
            >
              +
            </button>
            <textarea
              id="composer-input"
              class="composer__input"
              placeholder={inputPlaceholder}
              bind:value
              bind:this={textareaEl}
              rows="1"
              on:keydown={onKeydown}
              on:input={autoResize}
              on:paste={handleMediaPaste}
            ></textarea>
          </div>
          <input
            class="composer__file-input"
            type="file"
            accept="image/*,audio/*,application/pdf,.pdf,.mp3,.wav,.ogg,.webm,.m4a,.flac"
            multiple
            bind:this={mediaInputEl}
            on:change={handleMediaInputChange}
            disabled={disabled || isRunning || isCancelling || isConvertingPdf}
          />
          {#if isConvertingPdf}
            <p class="composer__media-status">Converting PDF pages...</p>
          {/if}
          {#if hasMediaAttachments || pdfPageAttachments.length > 0}
            <div class="composer__media-toolbar">
              <button
                type="button"
                class="composer__upload-trigger"
                on:click={clearMediaAttachments}
                disabled={disabled || isRunning || isCancelling || isConvertingPdf}
              >
                Clear media
              </button>
            </div>
          {/if}

          {#if imageAttachments.length > 0}
            <div class="composer__media-section">
              <p class="composer__media-title">Images ({imageAttachments.length})</p>
              <ul class="composer__media-list">
                {#each imageAttachments as item (item.id)}
                  <li class="composer__media-item">
                    <span class="composer__media-name">{item.name}</span>
                    <button
                      type="button"
                      class="composer__media-remove"
                      on:click={() => removeImageAttachment(item.id)}
                      disabled={disabled || isRunning || isCancelling}
                    >
                      Remove
                    </button>
                  </li>
                {/each}
              </ul>
            </div>
          {/if}

          {#if audioAttachments.length > 0}
            <div class="composer__media-section">
              <p class="composer__media-title">Audio ({audioAttachments.length})</p>
              <ul class="composer__media-list">
                {#each audioAttachments as item (item.id)}
                  <li class="composer__media-item">
                    <span class="composer__media-name">{item.name}</span>
                    <button
                      type="button"
                      class="composer__media-remove"
                      on:click={() => removeAudioAttachment(item.id)}
                      disabled={disabled || isRunning || isCancelling}
                    >
                      Remove
                    </button>
                  </li>
                {/each}
              </ul>
            </div>
          {/if}

          {#if pdfPageAttachments.length > 0}
            <div class="composer__media-section">
              <div class="composer__pdf-header">
                <p class="composer__media-title">
                  PDF pages selected {selectedPdfPageCount}/{MAX_SELECTED_PDF_PAGES}
                </p>
                <button
                  type="button"
                  class="composer__media-remove"
                  on:click={clearPdfAttachments}
                  disabled={disabled || isRunning || isCancelling || isConvertingPdf}
                >
                  Remove PDF
                </button>
              </div>
              <div class="composer__pdf-grid">
                {#each pdfPageAttachments as page (page.id)}
                  <button
                    type="button"
                    class="composer__pdf-page"
                    class:composer__pdf-page--selected={page.selected}
                    on:click={() => togglePdfPageSelection(page.id)}
                    disabled={(!page.selected && selectedPdfPageCount >= MAX_SELECTED_PDF_PAGES) || disabled || isRunning || isCancelling}
                    aria-pressed={page.selected}
                  >
                    <img src={page.dataUrl} alt={`${page.fileName} page ${page.pageNumber}`} loading="lazy" />
                    <span>Page {page.pageNumber}</span>
                  </button>
                {/each}
              </div>
            </div>
          {/if}
        </div>
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
    {#if mediaError}
      <p class="composer__error" role="alert">{mediaError}</p>
    {/if}
    {#if pdfSelectionRequiredError}
      <p class="composer__error" role="alert">{pdfSelectionRequiredError}</p>
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
    border-radius: calc(var(--radius-sm) - 2px);
    border: none;
    padding: var(--spacing-sm) var(--spacing-md);
    font: inherit;
    line-height: 1.4;
    color: var(--text-primary);
    background: #fff;
    caret-color: var(--color-uni-blue);
  }

  .composer__multimodal-fieldset {
    flex: 1;
    display: grid;
    gap: var(--spacing-xs);
  }

  .composer__multimodal-input-row {
    display: flex;
    align-items: flex-end;
    gap: var(--spacing-xs);
    border: 1px solid rgba(52, 74, 154, 0.25);
    border-radius: var(--radius-sm);
    background: #fff;
    padding: 6px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
  }

  .composer__multimodal-input-row--drag-over {
    border-color: var(--color-uni-blue);
    box-shadow: 0 0 0 2px rgba(52, 74, 154, 0.18);
    background: rgba(52, 74, 154, 0.03);
  }

  .composer__media-plus {
    width: 2.1rem;
    height: 2.1rem;
    border-radius: var(--radius-sm);
    border: 1px solid rgba(52, 74, 154, 0.28);
    background: var(--surface-base);
    color: var(--color-uni-blue);
    font: inherit;
    font-size: 1.2rem;
    font-weight: 700;
    line-height: 1;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    flex: 0 0 auto;
    transition: transform 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
  }

  .composer__media-plus:not(:disabled):hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 12px rgba(52, 74, 154, 0.16);
  }

  .composer__media-plus:disabled {
    cursor: not-allowed;
    opacity: 0.6;
    transform: none;
    box-shadow: none;
  }

  .composer__media-toolbar {
    display: flex;
    justify-content: flex-start;
  }

  .composer__media-status {
    margin: 0;
    font-size: 0.78rem;
    color: var(--text-subtle);
  }

  .composer__media-section {
    border: 1px solid rgba(52, 74, 154, 0.2);
    border-radius: var(--radius-sm);
    padding: var(--spacing-xs);
    background: rgba(52, 74, 154, 0.04);
    display: grid;
    gap: var(--spacing-xs);
  }

  .composer__media-title {
    margin: 0;
    font-size: 0.82rem;
    color: var(--text-primary);
    font-weight: 600;
  }

  .composer__media-list {
    margin: 0;
    padding: 0;
    list-style: none;
    display: grid;
    gap: 4px;
  }

  .composer__media-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--spacing-xs);
    border-radius: var(--radius-sm);
    background: #fff;
    padding: 0.3rem 0.45rem;
    border: 1px solid rgba(0, 0, 0, 0.08);
  }

  .composer__media-name {
    font-size: 0.78rem;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .composer__media-remove {
    border: 1px solid rgba(193, 0, 42, 0.2);
    color: var(--color-uni-red);
    background: rgba(193, 0, 42, 0.08);
    border-radius: var(--radius-sm);
    font-size: 0.75rem;
    padding: 0.2rem 0.45rem;
    cursor: pointer;
  }

  .composer__media-remove:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }

  .composer__pdf-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: var(--spacing-xs);
    flex-wrap: wrap;
  }

  .composer__pdf-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(96px, 1fr));
    gap: var(--spacing-xs);
  }

  .composer__pdf-page {
    border: 1px solid rgba(0, 0, 0, 0.14);
    border-radius: var(--radius-sm);
    background: #fff;
    padding: 4px;
    display: grid;
    gap: 4px;
    cursor: pointer;
    color: var(--text-primary);
    font-size: 0.72rem;
    text-align: left;
  }

  .composer__pdf-page img {
    width: 100%;
    height: 104px;
    object-fit: cover;
    border-radius: 4px;
    border: 1px solid rgba(0, 0, 0, 0.08);
  }

  .composer__pdf-page:disabled {
    opacity: 0.55;
    cursor: not-allowed;
  }

  .composer__pdf-page--selected {
    border-color: rgba(52, 74, 154, 0.85);
    box-shadow: inset 0 0 0 1px rgba(52, 74, 154, 0.5);
    background: rgba(52, 74, 154, 0.1);
  }

  .composer__input:focus {
    outline: none;
  }

  .composer__el-fieldset {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
    min-width: 0;
  }

  .composer__el-input-wrap {
    position: relative;
    width: 100%;
    background: #fff;
    border-radius: var(--radius-sm);
  }

  /* renders the same text as the textarea (which sits on top of it with a
     transparent background) to visualize the selected annotation window */
  .composer__el-backdrop {
    position: absolute;
    inset: 0;
    overflow: hidden;
    pointer-events: none;
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    padding: var(--spacing-sm) var(--spacing-md);
    font: inherit;
    line-height: 1.4;
    white-space: pre-wrap;
    overflow-wrap: break-word;
    color: transparent;
  }

  .composer__el-input {
    position: relative;
    display: block;
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
    background: transparent;
    caret-color: var(--color-uni-blue);
  }

  .composer__el-input:focus {
    outline: none;
    border-color: rgba(52, 74, 154, 0.4);
  }

  .composer__el-input:disabled {
    opacity: 0.6;
  }

  .composer__el-window {
    background: rgba(52, 74, 154, 0.22);
    color: transparent;
    border-radius: 2px;
  }

  .composer__el-window-bar {
    display: flex;
    flex-wrap: wrap;
    justify-content: space-between;
    align-items: center;
    gap: var(--spacing-xs);
  }

  .composer__el-window-status {
    margin: 0;
    font-size: 0.82rem;
    color: var(--text-subtle);
  }

  .composer__el-instructions {
    display: grid;
    gap: 4px;
  }

  .composer__el-instructions-label {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-subtle);
  }

  .composer__el-instructions-input {
    border: 1px solid rgba(0, 0, 0, 0.12);
    border-radius: var(--radius-sm);
    padding: 0.45rem 0.6rem;
    font: inherit;
    font-size: 0.9rem;
    color: var(--text-primary);
    background: #fff;
  }

  .composer__el-instructions-input:focus {
    outline: none;
    border-color: rgba(52, 74, 154, 0.4);
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
