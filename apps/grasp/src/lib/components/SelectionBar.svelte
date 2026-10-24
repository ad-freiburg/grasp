<script>
  import { createEventDispatcher, tick } from 'svelte';

  export let task = 'sparql-qa';
  export let tasks = [];
  export let knowledgeGraphs = [];
  export let disabled = false;
  export let className = '';
  export let compact = false;

  const dispatch = createEventDispatcher();

  let isOpen = false;
  let triggerEl;
  let menuEl;
  let menuStyle = '';

  $: activeTask = tasks.find((t) => t.id === task);

  // marks a label that does not fit, so its end fades out
  function overflowing(node) {
    const update = () =>
      node.classList.toggle('task-trigger__label--overflow', node.scrollWidth > node.clientWidth);
    const observer = new ResizeObserver(update);
    observer.observe(node);
    const mutations = new MutationObserver(update);
    mutations.observe(node, { childList: true, characterData: true, subtree: true });
    return {
      destroy() {
        observer.disconnect();
        mutations.disconnect();
      }
    };
  }

  function toggle() {
    if (disabled) return;
    if (isOpen) {
      close();
    } else {
      open();
    }
  }

  async function open() {
    isOpen = true;
    await tick();
    positionMenu();
    menuEl?.focus();
  }

  function close() {
    isOpen = false;
  }

  function positionMenu() {
    if (!triggerEl) return;
    const rect = triggerEl.getBoundingClientRect();
    menuStyle = `left: ${rect.left}px; bottom: ${window.innerHeight - rect.top + 4}px; min-width: ${rect.width}px;`;
  }

  function selectTask(id) {
    if (id !== task) {
      dispatch('taskchange', id);
    }
    close();
    triggerEl?.focus();
  }

  function handleMenuKeydown(event) {
    if (event.key === 'Escape') {
      event.preventDefault();
      close();
      triggerEl?.focus();
    } else if (event.key === 'ArrowDown') {
      event.preventDefault();
      focusSibling(1);
    } else if (event.key === 'ArrowUp') {
      event.preventDefault();
      focusSibling(-1);
    }
  }

  function handleTriggerKeydown(event) {
    if (event.key === 'ArrowDown' || event.key === 'ArrowUp') {
      event.preventDefault();
      open();
    }
  }

  function focusSibling(direction) {
    if (!menuEl) return;
    const items = [...menuEl.querySelectorAll('[role="option"]')];
    if (!items.length) return;
    const active = document.activeElement;
    const index = items.indexOf(active);
    let next = index + direction;
    if (next < 0) next = items.length - 1;
    if (next >= items.length) next = 0;
    items[next]?.focus();
  }

  function handleBackdropClick(event) {
    if (!triggerEl?.contains(event.target) && !menuEl?.contains(event.target)) {
      close();
    }
  }

  function toggleKg(id) {
    if (disabled) return;
    dispatch('kgchange', id);
  }
</script>

<svelte:window on:pointerdown={isOpen ? handleBackdropClick : undefined} />

<div
  class={`selection-bar ${className}`.trim()}
  class:selection-bar--compact={compact}
  aria-label="Knowledge graph selection"
>
  <div class="chip-row" class:chip-row--compact={compact}>
    <div class="task-select-container">
      <button
        type="button"
        class="task-trigger"
        bind:this={triggerEl}
        on:click={toggle}
        on:keydown={handleTriggerKeydown}
        {disabled}
        title={activeTask?.tooltip}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
      >
        <span class="task-trigger__label" use:overflowing>{activeTask?.name ?? task}</span>
        <span class="task-trigger__arrow" aria-hidden="true"></span>
      </button>
      {#if isOpen}
        <div
          class="task-menu"
          role="listbox"
          tabindex="-1"
          style={menuStyle}
          bind:this={menuEl}
          on:keydown={handleMenuKeydown}
          aria-label="Task"
        >
          {#each tasks as item (item.id)}
            <button
              type="button"
              class="task-menu__item"
              class:task-menu__item--active={item.id === task}
              role="option"
              aria-selected={item.id === task}
              title={item.tooltip}
              on:click={() => selectTask(item.id)}
            >
              {item.name}
            </button>
          {/each}
        </div>
      {/if}
    </div>

    {#each knowledgeGraphs as kg (kg.id)}
      <button
        type="button"
        class:chip--selected={kg.selected}
        class="chip"
        title={kg.selected ? `Exclude ${kg.id}` : `Include ${kg.id}`}
        aria-pressed={kg.selected}
        on:click={() => toggleKg(kg.id)}
        disabled={disabled}
      >
        <span class="chip__label">{kg.id}</span>
      </button>
    {/each}
  </div>
</div>

<style>
  .selection-bar {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-xs);
    justify-content: flex-start;
  }

  .selection-bar--compact {
    overflow: hidden;
  }

  .chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-xs);
    width: 100%;
    justify-content: flex-start;
    align-items: center;
  }

  .chip-row--compact {
    flex-wrap: nowrap;
    overflow-x: auto;
    gap: var(--spacing-xs);
    padding-bottom: 4px;
    scrollbar-width: none;
  }

  .chip-row--compact:hover,
  .chip-row--compact:focus-within {
    scrollbar-width: thin;
  }

  .chip-row::-webkit-scrollbar {
    height: 6px;
  }

  .chip-row--compact::-webkit-scrollbar {
    height: 0;
  }

  .chip-row--compact:hover::-webkit-scrollbar,
  .chip-row--compact:focus-within::-webkit-scrollbar {
    height: 6px;
  }

  .chip-row::-webkit-scrollbar-thumb {
    background: rgba(52, 74, 154, 0.3);
    border-radius: 999px;
  }

  .chip-row::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.05);
    border-radius: 999px;
  }

  .task-select-container {
    position: relative;
    margin-right: var(--spacing-xs);
    flex-shrink: 0;
    max-width: 12rem;
  }

  .task-trigger {
    appearance: none;
    border: 1px solid rgba(52, 74, 154, 0.28);
    border-radius: var(--radius-sm);
    background: rgba(52, 74, 154, 0.08);
    padding: 0.35rem 2rem 0.35rem 0.95rem;
    font-size: 0.85rem;
    line-height: 1.2;
    color: var(--color-uni-blue);
    cursor: pointer;
    transition: background 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    box-shadow: 0 4px 8px rgba(52, 74, 154, 0.12);
    font-family: inherit;
    font-weight: 600;
    min-width: 120px;
    max-width: 100%;
    text-align: left;
    position: relative;
    display: flex;
    align-items: center;
  }

  /* always one line like the chips, a long name fades out */
  .task-trigger__label {
    flex: 1;
    min-width: 0;
    overflow: hidden;
    white-space: nowrap;
  }

  :global(.task-trigger__label--overflow) {
    mask-image: linear-gradient(to right, #000 calc(100% - 1.5rem), transparent);
  }

  .task-trigger__arrow {
    position: absolute;
    right: 0.7rem;
    top: 50%;
    transform: translateY(-50%);
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid var(--color-uni-blue);
  }

  .task-trigger:not(:disabled):hover {
    box-shadow: 0 6px 12px rgba(52, 74, 154, 0.16);
  }

  .task-trigger:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }

  .task-trigger:focus {
    outline: none;
  }

  .task-menu {
    position: fixed;
    background: #fff;
    border: 1px solid rgba(52, 74, 154, 0.2);
    border-radius: var(--radius-sm);
    box-shadow: 0 8px 24px rgba(5, 17, 51, 0.15);
    z-index: 100;
    padding: 4px 0;
    outline: none;
  }

  .task-menu__item {
    appearance: none;
    display: block;
    width: 100%;
    border: none;
    background: none;
    padding: 0.5rem 0.95rem;
    font-size: 0.85rem;
    font-family: inherit;
    font-weight: 500;
    color: var(--text-primary);
    text-align: left;
    cursor: pointer;
    white-space: nowrap;
    transition: background 0.15s ease;
  }

  .task-menu__item:hover,
  .task-menu__item:focus {
    background: rgba(52, 74, 154, 0.08);
    outline: none;
  }

  .task-menu__item--active {
    color: var(--color-uni-blue);
    font-weight: 600;
  }

  .chip {
    appearance: none;
    border: 1px solid var(--border-default);
    border-radius: 999px;
    background: var(--surface-base);
    padding: 0.35rem 0.95rem;
    font-size: 0.85rem;
    line-height: 1.2;
    color: var(--text-primary);
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-xs);
    cursor: pointer;
    transition: background 0.2s ease, color 0.2s ease, border-color 0.2s ease;
    white-space: nowrap;
    scroll-snap-align: start;
    box-shadow: 0 6px 14px rgba(15, 15, 47, 0.08);
  }

  .chip--selected {
    background: var(--color-uni-blue);
    color: #fff;
    border-color: transparent;
  }

  .chip:not(:disabled):hover {
    box-shadow: none;
  }

  .chip:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }

  .visually-hidden {
    position: absolute;
    width: 1px;
    height: 1px;
    padding: 0;
    margin: -1px;
    overflow: hidden;
    clip: rect(0, 0, 0, 0);
    white-space: nowrap;
    border: 0;
  }
</style>
