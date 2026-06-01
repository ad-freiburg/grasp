<script>
  import { onMount, onDestroy } from 'svelte';
  import { wsEndpoint, kgEndpoint, COPYRIGHT, MATERIAL_URL } from '../constants.js';
  import StepCard from './StepCard.svelte';
  import OutputCard from './OutputCard.svelte';
  import SparqlBlock from './SparqlBlock.svelte';
  import examples from '../examples.json';

  let question = '';
  let connectionStatus = 'initial';
  let statusMessage = '';
  let running = false;
  let cancelling = false;
  let socket;
  let kg = '';
  let textareaEl;
  let examplesOpen = false;
  let examplesButtonEl;

  // generation state
  let steps = [];
  let output = null;
  let error = null;
  let pendingCancelSignal = false;
  let expanded = false;

  $: connected = connectionStatus === 'connected';
  $: canSubmit = connected && !running && question.trim().length > 0;
  $: hasSteps = steps.length > 0;
  $: hasOutput = output !== null;
  $: hasResult = hasSteps || hasOutput || error !== null;
  $: question, autoResize();

  // Group selection steps by skeleton for display
  $: groupedSteps = (() => {
    const result = [];
    const skeletonsStep = steps.find(s => s.type === 'skeletons');
    if (skeletonsStep) {
      result.push(skeletonsStep);
    }

    // Group selections by skeleton number
    const selectionsByskeleton = new Map();
    steps.filter(s => s.type === 'selection').forEach(step => {
      const skeletonId = step.skeleton;
      if (!selectionsByskeleton.has(skeletonId)) {
        selectionsByskeleton.set(skeletonId, []);
      }
      selectionsByskeleton.get(skeletonId).push(step.selection);
    });

    // Add grouped selections with skeleton code
    for (const [skeletonId, selections] of selectionsByskeleton.entries()) {
      const skeletonCode = skeletonsStep?.skeletons?.[skeletonId] ?? null;
      result.push({
        type: 'skeleton-selections',
        skeleton: skeletonId,
        selections,
        skeletonCode
      });
    }

    return result;
  })();

  onMount(async () => {
    await initialize();
    autoResize();
    if (typeof document !== 'undefined') {
      document.addEventListener('click', handleClickOutside);
    }
  });

  onDestroy(() => {
    cleanupSocket();
    if (typeof document !== 'undefined') {
      document.removeEventListener('click', handleClickOutside);
    }
  });

  async function initialize() {
    try {
      await loadKnowledgeGraph();
      await openConnection();
    } catch (err) {
      console.error('Failed to initialize', err);
      statusMessage = 'Failed to initialize. Please check your connection and reload.';
    }
  }

  async function loadKnowledgeGraph() {
    const response = await fetch(kgEndpoint());
    if (!response.ok) throw new Error('Failed to load knowledge graph info');
    const kgs = await response.json();
    if (Array.isArray(kgs) && kgs.length > 0) {
      kg = kgs[0];
    }
  }

  async function openConnection() {
    cleanupSocket();
    connectionStatus = 'connecting';
    return new Promise((resolve, reject) => {
      try {
        socket = new WebSocket(wsEndpoint());
      } catch (err) {
        connectionStatus = 'error';
        return reject(err);
      }

      socket.addEventListener('open', () => {
        connectionStatus = 'connected';
        statusMessage = '';
        resolve();
      });

      socket.addEventListener('message', handleSocketMessage);
      socket.addEventListener('close', handleSocketClose);
      socket.addEventListener('error', () => {
        connectionStatus = 'error';
        statusMessage = 'WebSocket error occurred.';
        reject(new Error('WebSocket error'));
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
    statusMessage = event?.reason || 'Connection lost. Please reload to reconnect.';
  }

  function handleSocketMessage(event) {
    try {
      const payload = JSON.parse(event.data);

      if (payload.error && !payload.type) {
        statusMessage = payload.error;
        running = false;
        cancelling = false;
        sendAck();
        return;
      }

      if (payload.cancelled) {
        steps = [];
        output = null;
        expanded = false;
        running = false;
        cancelling = false;
        return;
      }

      if (!payload.type) {
        sendAck();
        return;
      }

      if (payload.type === 'skeletons') {
        steps = [...steps, { type: 'skeletons', skeletons: payload.skeletons }];
      } else if (payload.type === 'selection') {
        steps = [...steps, {
          type: 'selection',
          skeleton: payload.skeleton,
          selection: payload.selection,
        }];
      } else if (payload.type === 'output') {
        output = payload;
        running = false;
        cancelling = false;
        pendingCancelSignal = false;
      }

      sendAck();
    } catch (err) {
      console.error('Failed to handle message', err);
    }
  }

  function sendAck() {
    if (!socket || socket.readyState !== WebSocket.OPEN) return;
    const payload = { received: true };
    if (pendingCancelSignal) {
      payload.cancel = true;
      pendingCancelSignal = false;
    }
    socket.send(JSON.stringify(payload));
  }

  function handleSubmit() {
    if (!canSubmit) return;

    statusMessage = '';
    error = null;
    steps = [];
    output = null;
    running = true;

    try {
      socket?.send(JSON.stringify({ question: question.trim() }));
    } catch (err) {
      statusMessage = 'Failed to send request.';
      running = false;
    }
  }

  function handleCancel() {
    if (!connected) return;
    cancelling = true;
    pendingCancelSignal = true;
  }

  function handleReset() {
    question = '';
    steps = [];
    output = null;
    error = null;
    statusMessage = '';
    expanded = false;
  }

  function toggleExpanded() {
    expanded = !expanded;
  }

  function handleKeydown(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      handleSubmit();
    }
  }

  function autoResize() {
    if (!textareaEl) return;
    textareaEl.style.height = 'auto';
    const newHeight = Math.min(textareaEl.scrollHeight, 200); // max 200px
    textareaEl.style.height = `${newHeight}px`;
  }

  function toggleExamples() {
    examplesOpen = !examplesOpen;
  }

  function handleExampleSelect(index) {
    if (index >= 0 && index < examples.length) {
      question = examples[index].question;
      autoResize();
      examplesOpen = false;
    }
  }

  function handleClickOutside(event) {
    if (examplesOpen && examplesButtonEl && !examplesButtonEl.contains(event.target)) {
      examplesOpen = false;
    }
  }
</script>

<section class="app-shell">
  <footer class="footer">
    <p>&copy; {new Date().getFullYear()} {COPYRIGHT}</p>
    <nav class="footer-links">
      <a href="evaluate">Evaluation</a>
      <a href={MATERIAL_URL}>Material</a>
    </nav>
  </footer>

  <div class="shell-content" class:shell-content--empty={!hasResult}>
    <header class="header">
      <div class="title-stack">
        <h1>GRISP</h1>
        <p>Guided Recurrent IRI Selection over SPARQL Skeletons</p>
      </div>
    </header>

    <main class="main-content" class:main-content--centered={!hasResult}>
    <!-- Input -->
    <div class="input-section" class:input-section--running={running}>
      <div class="input-row">
        <textarea
          class="question-input"
          placeholder={connected ? (kg ? `Ask a question over ${kg}...` : 'Ask a question...') : 'Connecting...'}
          bind:value={question}
          bind:this={textareaEl}
          on:keydown={handleKeydown}
          on:input={autoResize}
          rows="1"
          disabled={!connected || running}
        ></textarea>
        <div class="input-actions">
          {#if examples.length > 0}
            <div class="examples-dropdown" bind:this={examplesButtonEl}>
              <button
                type="button"
                class="icon-button icon-button--secondary"
                on:click={toggleExamples}
                disabled={!connected || running}
                aria-label="Examples"
                title="Example questions"
              >
                <span aria-hidden="true">☰</span>
              </button>
              {#if examplesOpen}
                <div class="examples-menu">
                  {#each examples as example, i (i)}
                    <button
                      type="button"
                      class="example-item"
                      on:click={() => handleExampleSelect(i)}
                    >
                      <div class="example-label">{example.label}</div>
                      <div class="example-question">{example.question}</div>
                    </button>
                  {/each}
                </div>
              {/if}
            </div>
          {/if}
          <button
            type="button"
            class="icon-button icon-button--primary"
            on:click={handleSubmit}
            disabled={!canSubmit}
            aria-label="Run"
            title="Run"
          >
            <span class="paperplane-icon" aria-hidden="true">➤</span>
          </button>
          {#if running}
            <button
              type="button"
              class="icon-button icon-button--danger"
              class:icon-button--cancelling={cancelling}
              on:click={handleCancel}
              disabled={cancelling}
              aria-label={cancelling ? 'Cancellation in progress' : 'Cancel'}
              title={cancelling ? 'Cancellation in progress' : 'Cancel'}
            >
              {#if cancelling}
                <span class="cancel-spinner" aria-hidden="true"></span>
              {:else}
                <span class="cancel-icon" aria-hidden="true">✖</span>
              {/if}
            </button>
          {/if}
          {#if hasResult && !running}
            <button
              type="button"
              class="icon-button icon-button--clear"
              on:click={handleReset}
              disabled={false}
              aria-label="Clear"
              title="Clear"
            >
              <span aria-hidden="true">↺</span>
            </button>
          {/if}
        </div>
      </div>

      {#if statusMessage}
        <div class="alert-box" role="alert">
          <div class="alert-text">
            <strong>Connection issue</strong>
            <span>{statusMessage}</span>
          </div>
          {#if connectionStatus === 'disconnected' || connectionStatus === 'error'}
            <button
              type="button"
              class="alert-button"
              on:click={() => window.location.reload()}
            >
              Reload page
            </button>
          {/if}
        </div>
      {/if}
    </div>

    <!-- Progress -->
    {#if running && !hasOutput}
      <div class="progress-section">
        <div class="progress-bar">
          <span class="spinner" aria-hidden="true"></span>
          <span class="progress-text">
            {#if steps.length === 0}
              Generating skeletons...
            {:else}
              {@const latestSelection = steps.filter(s => s.type === 'selection').slice(-1)[0]}
              {#if latestSelection}
                Selecting knowledge graph items for skeleton {latestSelection.skeleton + 1}...
              {:else}
                Starting selection...
              {/if}
            {/if}
          </span>
        </div>
      </div>
    {/if}

    <!-- Output -->
    {#if hasOutput}
      <OutputCard {output} />
    {/if}

    <!-- Show details button -->
    {#if hasSteps}
      <div class="toggle-section">
        <button class="toggle-button" on:click={toggleExpanded}>
          <span class="toggle-arrow">{expanded ? '▲' : '▼'}</span>
          <span>{expanded ? 'Hide details' : 'Show details'}</span>
        </button>
      </div>
    {/if}

    <!-- Intermediate steps (shown when expanded) -->
    {#if hasSteps && expanded}
      {#each groupedSteps as step, i (i)}
        {#if step.type === 'skeletons'}
          <div class="steps-section">
            <div class="steps-header">
              Generated Skeletons ({step.skeletons?.length ?? 0})
            </div>
            <div class="steps-list">
              <StepCard {step} index={i} />
            </div>
          </div>
        {:else if step.type === 'skeleton-selections'}
          <div class="steps-section">
            <div class="steps-header">
              Skeleton #{step.skeleton + 1} - Selection
            </div>
            <div class="steps-list">
              <StepCard {step} index={i} />
            </div>
          </div>
        {/if}
      {/each}
    {/if}
    </main>
  </div>
</section>

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

  .footer {
    text-align: center;
    padding: 0.25rem 0;
    color: var(--text-subtle);
    font-size: 0.75rem;
    line-height: 1.3;
    display: grid;
    justify-items: center;
    gap: var(--spacing-xs);
  }

  .footer p {
    margin: 0;
  }

  .footer-links {
    display: flex;
    gap: var(--spacing-sm);
    font-size: 0.85rem;
    flex-wrap: wrap;
    justify-content: center;
  }

  .footer-links a {
    color: var(--color-uni-blue);
    text-decoration: underline;
  }

  .footer-links a:hover,
  .footer-links a:focus-visible {
    text-decoration: none;
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

  .header {
    display: flex;
    justify-content: center;
    text-align: center;
    padding: var(--spacing-lg) 0 var(--spacing-md);
  }

  .title-stack {
    display: grid;
    gap: var(--spacing-sm);
  }

  .title-stack h1 {
    font-size: clamp(2.25rem, 3.5vw, 3rem);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--color-uni-blue);
    margin: 0;
  }

  .title-stack p {
    margin: 0;
    color: var(--text-subtle);
    font-size: 1rem;
  }

  .main-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    gap: var(--spacing-lg);
    padding-bottom: var(--spacing-lg);
  }

  .main-content--centered {
    flex: 0 0 auto;
    display: flex;
    align-items: stretch;
    justify-content: center;
    flex-direction: column;
    gap: var(--spacing-lg);
    width: 100%;
  }

  .main-content--centered .input-section {
    width: 100%;
    max-width: 100%;
  }

  .input-section {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
    background: var(--surface-base);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    padding: var(--spacing-lg);
    box-shadow: var(--shadow-sm);
    position: relative;
    overflow: visible;
  }

  .input-section::after {
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

  .input-section.input-section--running::after {
    opacity: 1;
    animation: progress-bar 1.2s linear infinite;
  }

  @keyframes progress-bar {
    from {
      background-position: 0% 0;
    }
    to {
      background-position: 200% 0;
    }
  }

  .examples-dropdown {
    position: relative;
  }

  .examples-menu {
    position: absolute;
    top: calc(100% + 4px);
    right: 0;
    min-width: 280px;
    max-width: 400px;
    max-height: 40vh;
    overflow-y: auto;
    background: var(--surface-base);
    border: 1px solid rgba(52, 74, 154, 0.25);
    border-radius: var(--radius-sm);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    z-index: 1000;
  }

  .example-item {
    display: block;
    width: 100%;
    padding: var(--spacing-sm) var(--spacing-md);
    text-align: left;
    border: none;
    background: none;
    cursor: pointer;
    transition: background 0.15s ease;
    border-bottom: 1px solid rgba(0, 0, 0, 0.06);
  }

  .example-item:last-child {
    border-bottom: none;
  }

  .example-item:hover {
    background: rgba(52, 74, 154, 0.08);
  }

  .example-item:focus {
    outline: none;
    background: rgba(52, 74, 154, 0.12);
  }

  .example-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--color-uni-blue);
    margin-bottom: 0.25rem;
  }

  .example-question {
    font-size: 0.8rem;
    color: var(--text-subtle);
    line-height: 1.3;
  }

  .input-row {
    display: flex;
    gap: var(--spacing-sm);
    align-items: stretch;
  }

  .question-input {
    flex: 1;
    resize: none;
    min-height: 2.5rem;
    max-height: 200px;
    padding: var(--spacing-sm) var(--spacing-md);
    border: 1px solid rgba(0, 0, 0, 0.12);
    border-radius: var(--radius-sm);
    font-size: 0.95rem;
    font-family: inherit;
    background: #fff;
    color: var(--text-primary);
    outline: none;
    caret-color: var(--color-uni-blue);
    line-height: 1.4;
    overflow-y: auto;
  }

  .question-input:focus {
    outline: none;
  }

  .question-input:disabled {
    opacity: 0.6;
    cursor: not-allowed;
  }

  .input-actions {
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
    background: var(--color-uni-red);
    color: #fff;
    box-shadow: 0 4px 8px rgba(193, 0, 42, 0.18);
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

  .icon-button--secondary {
    background: rgba(52, 74, 154, 0.08);
    color: var(--color-uni-blue);
    border: 1px solid rgba(52, 74, 154, 0.2);
  }

  .icon-button--secondary:not(:disabled):hover {
    background: rgba(52, 74, 154, 0.12);
  }

  .icon-button--clear {
    background: rgba(52, 74, 154, 0.12);
    color: var(--color-uni-blue);
    border: 1px solid rgba(52, 74, 154, 0.18);
    box-shadow: 0 4px 8px rgba(52, 74, 154, 0.16);
  }

  .icon-button:not(:disabled):hover {
    transform: translateY(-1px);
  }

  .cancel-spinner {
    width: 1.05rem;
    height: 1.05rem;
    border-radius: 50%;
    border: 2px solid rgba(193, 0, 42, 0.28);
    border-top-color: var(--color-uni-red);
    animation: cancel-spin 0.7s linear infinite;
  }

  @keyframes cancel-spin {
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

  .alert-box {
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

  .alert-text {
    display: grid;
    gap: 2px;
  }

  .alert-text strong {
    font-size: 0.95rem;
  }

  .alert-text span {
    font-size: 0.85rem;
    color: var(--text-primary);
  }

  .alert-button {
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

  .alert-button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 14px rgba(52, 74, 154, 0.2);
  }

  .alert-button:focus-visible {
    outline: 2px solid rgba(52, 74, 154, 0.4);
    outline-offset: 2px;
  }

  .progress-section {
    display: flex;
    justify-content: center;
  }

  .progress-bar {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-sm);
    padding: var(--spacing-sm) var(--spacing-md);
    background: rgba(52, 74, 154, 0.06);
    border-radius: var(--radius-sm);
    color: var(--color-uni-blue);
    font-weight: 500;
    font-size: 0.9rem;
    width: fit-content;
    position: relative;
    overflow: hidden;
  }

  .progress-bar::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(
      90deg,
      rgba(52, 74, 154, 0) 0%,
      rgba(52, 74, 154, 0.15) 50%,
      rgba(52, 74, 154, 0) 100%
    );
    background-size: 200% 100%;
    animation: pulsate 2s linear infinite;
  }

  @keyframes pulsate {
    0% {
      background-position: 200% 0;
    }
    100% {
      background-position: -200% 0;
    }
  }

  .progress-text {
    min-width: 0;
  }

  .spinner {
    width: 1rem;
    height: 1rem;
    border-radius: 50%;
    border: 2.5px solid rgba(52, 74, 154, 0.25);
    border-top-color: var(--color-uni-blue);
    animation: spin 0.9s linear infinite;
    flex-shrink: 0;
  }

  @keyframes spin {
    from { transform: rotate(0deg); }
    to { transform: rotate(360deg); }
  }

  .toggle-section {
    display: flex;
    justify-content: center;
  }

  .toggle-button {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-xs);
    padding: var(--spacing-xs) var(--spacing-sm);
    background: transparent;
    border: none;
    border-radius: var(--radius-sm);
    color: var(--text-subtle);
    font-size: 0.8rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s ease, color 0.2s ease;
  }

  .toggle-button:hover {
    background: rgba(0, 0, 0, 0.04);
    color: var(--text-primary);
  }

  .toggle-button:focus {
    outline: none;
    background: rgba(0, 0, 0, 0.06);
  }

  .toggle-arrow {
    font-size: 0.7rem;
    opacity: 0.7;
  }

  .steps-section {
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    background: var(--surface-base);
    overflow: hidden;
    box-shadow: var(--shadow-sm);
  }

  .steps-header {
    padding: var(--spacing-md) var(--spacing-lg);
    font-weight: 600;
    font-size: 0.9rem;
    color: var(--text-subtle);
    background: rgba(52, 74, 154, 0.02);
    border-bottom: 1px solid var(--border-default);
  }

  .steps-list {
    display: grid;
    gap: 1px;
    background: var(--border-default);
  }

</style>
