<script>
  import { onMount, onDestroy, afterUpdate, tick } from 'svelte';
  import { fade, scale } from 'svelte/transition';
  import { cubicOut } from 'svelte/easing';
  import InputMessage from './history/InputMessage.svelte';
  import SystemMessage from './history/SystemMessage.svelte';
  import FeedbackMessage from './history/FeedbackMessage.svelte';
  import ReasoningMessage from './history/ReasoningMessage.svelte';
  import ToolMessage from './history/ToolMessage.svelte';
  import OutputMessage from './history/OutputMessage.svelte';
  import UnknownMessage from './history/UnknownMessage.svelte';

  export let histories = [];
  export let running = false;
  export let cancelling = false;
  export let composerOffset = 0;
  export let shareConversation = null;
  export let selectedKgs = [];

  let listEl;
  let stickToBottom = true;
  let previousCount = 0;
  let expandedHistories = new Set();
  let recentToolMessages = []; // Array of { id, message }
  let nextMessageId = 0;
  let lastProcessedMessageIndex = -1;
  const MAX_CONCURRENT_MESSAGES = 5;

  $: showKgChip = selectedKgs.length > 1;

  // Track recent tool messages and show them briefly
  $: if (running && histories.length > 0) {
    const currentHistory = histories[histories.length - 1];
    const currentIndex = currentHistory.length - 1;
    const lastMsg = currentHistory[currentIndex];

    // Only process if this is a new message we haven't seen
    if (lastMsg?.type === 'tool' && currentIndex > lastProcessedMessageIndex) {
      lastProcessedMessageIndex = currentIndex;
      addToolMessage(lastMsg);
    }
  }

  // Reset when not running
  $: if (!running) {
    lastProcessedMessageIndex = -1;
    recentToolMessages = [];
  }

  function addToolMessage(message) {
    const id = nextMessageId++;

    // Add to array
    recentToolMessages = [...recentToolMessages, { id, message }];

    // If we exceed max, remove oldest
    if (recentToolMessages.length > MAX_CONCURRENT_MESSAGES) {
      recentToolMessages = recentToolMessages.slice(1);
    }

    // Set timeout to remove this specific message after 2.5 seconds
    setTimeout(() => {
      recentToolMessages = recentToolMessages.filter(item => item.id !== id);
    }, 2500);
  }

  onDestroy(() => {
    // Timeouts will clean themselves up
    recentToolMessages = [];
  });

  function toggleExpanded(historyIndex) {
    if (expandedHistories.has(historyIndex)) {
      expandedHistories.delete(historyIndex);
    } else {
      expandedHistories.add(historyIndex);
    }
    expandedHistories = expandedHistories;
  }

  function getCompactStatus(message) {
    if (!message) return { kg: null, status: '' };

    if (message.type === 'tool') {
      const toolName = message.name || '';
      const args = message.args || {};
      const kg = args.kg || null;

      let status = '';

      if (toolName === 'search_entity') {
        const query = args.query || '';
        status = query
          ? `Searching for entity "${query}"...`
          : 'Searching for entity...';
      } else if (toolName === 'search_property') {
        const query = args.query || '';
        status = query
          ? `Searching for property "${query}"...`
          : 'Searching for property...';
      } else if (toolName === 'search_property_of_entity') {
        const entity = args.entity || 'entity';
        const query = args.query || '';
        status = query
          ? `Searching for property "${query}" of ${entity}...`
          : `Searching for properties of ${entity}...`;
      } else if (toolName === 'search_object_of_property') {
        const property = args.property || 'property';
        const query = args.query || '';
        status = query
          ? `Searching for object "${query}" of ${property}...`
          : `Searching for objects of ${property}...`;
      } else if (toolName === 'search_with_constraints' || toolName === 'search') {
        const query = args.query || '';
        const position = args.position || 'item';
        const constraints = args.constraints || {};
        const hasConstraints = Object.values(constraints).some(v => v !== null);

        if (hasConstraints) {
          status = query
            ? `Searching for ${position} "${query}" with constraints...`
            : `Searching for ${position} with constraints...`;
        } else {
          status = query
            ? `Searching for ${position} "${query}"...`
            : `Searching for ${position}...`;
        }
      } else if (toolName === 'search_with_filter') {
        const query = args.query || '';
        status = query
          ? `Searching for "${query}" with SPARQL query constraint...`
          : 'Searching with SPARQL query constraint...';
      } else if (toolName === 'execute') {
        status = 'Executing SPARQL query...';
      } else if (toolName === 'list') {
        const hasConstraints = [args.subject, args.property, args.object].some(
          v => v !== null && v !== undefined
        );
        status = hasConstraints
          ? 'Listing triples with constraints...'
          : 'Listing triples...';
      } else if (toolName === 'annotate') {
        const row = args.row;
        const col = args.column;
        const words = args.words_to_be_annotated;
        if (typeof words === 'string' && words) {
          status = `Annotating "${words}"...`;
        } else {
          status = row !== undefined && col !== undefined
            ? `Annotating cell (${row}, ${col})...`
            : 'Annotating...';
        }
      } else if (toolName === 'delete_annotation') {
        const row = args.row;
        const col = args.column;
        const words = args.words_to_be_annotated;
        if (typeof words === 'string' && words) {
          status = `Deleting annotation of "${words}"...`;
        } else {
          status = row !== undefined && col !== undefined
            ? `Deleting annotation from (${row}, ${col})...`
            : 'Deleting annotation...';
        }
      } else if (toolName === 'show_annotations' || toolName === 'show_current_annotations') {
        status = 'Showing annotations...';
      } else if (toolName === 'finalize') {
        status = 'Finalizing annotations...';
      } else if (toolName === 'stop') {
        status = 'Finalizing annotations...';
      } else if (toolName === 'answer') {
        status = 'Providing final answer...';
      } else if (toolName === 'cancel') {
        status = 'Cancelling query generation...';
      } else if (toolName === 'get_random_examples') {
        const kg = args.kg;
        status = kg ? `Finding examples for ${kg}...` : 'Finding examples...';
      } else if (toolName === 'search_example') {
        const kg = args.kg;
        status = kg ? `Finding similar examples for ${kg}...` : 'Finding similar examples...';
      } else if (toolName === 'search_literal') {
        const query = args.query || '';
        status = query ? `Searching for literal "${query}"...` : 'Searching for literal...';
      } else if (toolName === 'search_shape') {
        const query = args.query || '';
        status = query ? `Searching for shapes matching "${query}"...` : 'Searching for shapes...';
      } else if (toolName === 'get_shape') {
        const iri = args.iri || 'IRI';
        status = `Fetching shape for ${iri}...`;
      } else {
        status = `Calling ${toolName}...`;
      }

      return { kg, status };
    }

    if (message.type === 'model') {
      return { kg: null, status: 'Thinking...' };
    }

    return { kg: null, status: '' };
  }

  $: items = histories
    .map((history, historyIndex) =>
      history.map((message, index) => ({
        key: `${historyIndex}-${index}-${message?.type ?? 'message'}`,
        message,
        historyIndex
      }))
    )
    .flat();

  $: isEmpty = items.length === 0;
  $: paddedComposerOffset = Math.max(0, composerOffset);

  function handleScroll() {
    if (!listEl) return;
    const { scrollTop, scrollHeight, clientHeight } = listEl;
    const distanceFromBottom = scrollHeight - (scrollTop + clientHeight);
    const nearBottom = distanceFromBottom < 120;
    stickToBottom = nearBottom;
  }

  function scrollToTop() {
    listEl?.scrollTo({ top: 0, behavior: 'smooth' });
  }

  function scrollToBottom() {
    if (!listEl) return;
    stickToBottom = true;
    listEl.scrollTo({ top: listEl.scrollHeight, behavior: 'smooth' });
  }

  onMount(async () => {
    previousCount = items.length;
    if (items.length) {
      await tick();
      scrollToBottom();
    }
  });

  afterUpdate(() => {
    if (!listEl) return;
    const added = items.length > previousCount;
    previousCount = items.length;
    if (added && stickToBottom) {
      listEl.scrollTo({ top: listEl.scrollHeight, behavior: 'smooth' });
    }
  });
</script>

<section class="conversation">
  {#if isEmpty}
    {#if running}
      <div class="empty-state">
        <div class="empty-state__progress">
          <span class="spinner" aria-hidden="true"></span>
          <span>{cancelling ? 'Cancelling…' : 'Waiting for response…'}</span>
        </div>
      </div>
    {/if}
  {:else}
    <ul
      class="history"
      bind:this={listEl}
      on:scroll={handleScroll}
      style={`--composer-offset:${paddedComposerOffset}px;`}
    >
      {#each histories as history, historyIndex (historyIndex)}
        {@const isExpanded = expandedHistories.has(historyIndex)}
        {@const isCurrentAndRunning = running && historyIndex === histories.length - 1}
        {@const hasOutput = history.some(m => m?.type === 'output')}
        {@const isCompleted = hasOutput && !isCurrentAndRunning}

        <!-- Input message -->
        {#if history[0]?.type === 'input'}
          <li class="history-item">
            <InputMessage message={history[0]} />
          </li>
        {/if}

        <!-- Compact status (shown when collapsed and currently running) -->
        {#if !isExpanded && isCurrentAndRunning && !hasOutput}
          <li class="history-item">
            <div class="compact-status-row">
              <!-- Base "Thinking..." status -->
              <div class="compact-status">
                <span class="spinner" aria-hidden="true"></span>
                <span>Thinking...</span>
              </div>

              <!-- Recent function calls -->
              <div class="tool-messages">
                {#each recentToolMessages as item (item.id)}
                  {@const statusData = getCompactStatus(item.message)}
                  <div
                    class="tool-message"
                    in:fade={{ duration: 200, easing: cubicOut }}
                    out:fade={{ duration: 200, easing: cubicOut }}
                  >
                    {#if showKgChip && statusData.kg}
                      <span class="kg-chip">{statusData.kg}</span>
                    {/if}
                    <span>{statusData.status}</span>
                  </div>
                {/each}
              </div>
            </div>
          </li>
        {/if}

        <!-- Show details button (shown when collapsed and there are messages to show) -->
        {#if !isExpanded && history.length > 1}
          <li class="history-item">
            <button class="toggle-button" on:click={() => toggleExpanded(historyIndex)}>
              <span class="toggle-arrow">▼</span>
              <span>Show details</span>
            </button>
          </li>
        {/if}

        <!-- Full messages (shown when expanded) -->
        {#if isExpanded}
          {#each history.slice(1) as message, idx (idx)}
            {#if message?.type !== 'output'}
              <li class="history-item">
                {#if message?.type === 'system'}
                  <SystemMessage {message} />
                {:else if message?.type === 'feedback'}
                  <FeedbackMessage {message} />
                {:else if message?.type === 'model'}
                  <ReasoningMessage {message} />
                {:else if message?.type === 'tool'}
                  <ToolMessage {message} />
                {:else}
                  <UnknownMessage {message} />
                {/if}
              </li>
            {/if}
          {/each}

          <!-- Hide details button (shown after full messages when expanded) -->
          <li class="history-item">
            <button class="toggle-button" on:click={() => toggleExpanded(historyIndex)}>
              <span class="toggle-arrow">▲</span>
              <span>Hide details</span>
            </button>
          </li>
        {/if}

        <!-- Output (always shown when available) -->
        {#if hasOutput}
          {#each history.slice(1) as message, idx (idx)}
            {#if message?.type === 'output'}
              <li class="history-item">
                <OutputMessage
                  {message}
                  {shareConversation}
                  shareDisabled={Boolean(message?.shareLocked)}
                />
              </li>
            {/if}
          {/each}
        {/if}
      {/each}
    </ul>
  {/if}
</section>

<style>
  .conversation {
    flex: 1;
    overflow: hidden;
    position: relative;
    display: flex;
    background: transparent;
  }

  .empty-state {
    margin: auto;
    padding: var(--spacing-xl);
    display: flex;
    align-items: center;
    justify-content: center;
  }

  .empty-state__progress {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-sm);
    justify-content: center;
    color: var(--color-uni-blue);
    font-weight: 600;
  }

  .history {
    list-style: none;
    margin: 0;
    padding: var(--spacing-md) 0 var(--composer-offset, 0px);
    width: 100%;
    overflow-y: auto;
    display: grid;
    gap: var(--spacing-sm);
    grid-template-columns: minmax(0, 1fr);
    align-content: start;
    justify-content: stretch;
    flex: 1 1 auto;
  }

  .history-item {
    list-style: none;
    padding: 0;
  }

  .compact-status-row {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    flex-wrap: wrap;
  }

  .compact-status {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-sm);
    padding: var(--spacing-sm) var(--spacing-md);
    background: rgba(52, 74, 154, 0.06);
    border-radius: var(--radius-sm);
    color: var(--color-uni-blue);
    font-size: 0.9rem;
    font-weight: 500;
    width: fit-content;
    position: relative;
    overflow: hidden;
  }

  .compact-status::before {
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

  .tool-messages {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-xs);
  }

  .tool-message {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-xs);
    padding: var(--spacing-xs) var(--spacing-sm);
    background: rgba(190, 170, 60, 0.08);
    border: 1px solid rgba(190, 170, 60, 0.3);
    border-radius: var(--radius-sm);
    color: var(--color-uni-yellow);
    font-size: 0.85rem;
    font-weight: 500;
  }

  .kg-chip {
    display: inline-flex;
    align-items: center;
    padding: 0.1rem 0.4rem;
    background: rgba(190, 170, 60, 0.2);
    color: var(--color-uni-yellow);
    border-radius: var(--radius-sm);
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.02em;
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

  @media (max-width: 720px) {
    .history {
      padding: var(--spacing-lg) 0 var(--composer-offset, 0px);
      align-content: start;
    }
  }

  .spinner {
    width: 1rem;
    height: 1rem;
    border-radius: 999px;
    border: 3px solid rgba(52, 74, 154, 0.25);
    border-top-color: var(--color-uni-blue);
    animation: spin 0.9s linear infinite;
  }

  @keyframes spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }
</style>
