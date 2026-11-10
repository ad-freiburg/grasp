<script>
  import { createEventDispatcher } from 'svelte';

  export let code = '';
  export let qleverLink = null;
  export let label = '';
  // optional marks in the code, each { start, end, item } with character
  // offsets and an item whose status is used as css class suffix
  export let marks = null;

  const dispatch = createEventDispatcher();

  function splitCode(code, marks) {
    const sorted = [...(marks ?? [])].sort((a, b) => a.start - b.start);
    const segments = [];
    let last = 0;
    for (const { start, end, item } of sorted) {
      if (start < last) continue;
      if (start > last) segments.push({ text: code.slice(last, start) });
      segments.push({ text: code.slice(start, end), item });
      last = end;
    }
    if (last < code.length) segments.push({ text: code.slice(last) });
    return segments;
  }

  $: segments = splitCode(code, marks);

  function handleKeydown(event, item) {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      dispatch('itemclick', { item, target: event.currentTarget });
    }
  }
</script>

{#if code}
  <div class="sparql-block">
    {#if label}
      <div class="sparql-block__label">{label}</div>
    {/if}
    {#if qleverLink}
      <div class="sparql-block__actions">
        <a
          class="sparql-link"
          href={qleverLink}
          target="_blank"
          rel="noreferrer"
        >
          Execute on QLever
        </a>
      </div>
    {/if}
    <div class="sparql-block__code">
      <!-- no whitespace between the segments, it would show up in the code -->
      <pre><code>{#each segments as segment}{#if segment.item}<span
              class={`item item--${segment.item.status}`}
              role="button"
              tabindex="0"
              on:pointerenter={(event) => event.pointerType === 'mouse' && dispatch('itemhover', { item: segment.item, target: event.currentTarget })}
              on:pointerleave={(event) => event.pointerType === 'mouse' && dispatch('itemleave')}
              on:click={(event) => dispatch('itemclick', { item: segment.item, target: event.currentTarget })}
              on:keydown={(event) => handleKeydown(event, segment.item)}
            >{segment.text}</span>{:else}{segment.text}{/if}{/each}</code></pre>
    </div>
  </div>
{/if}

<style>
  .sparql-block {
    position: relative;
    border: 1px solid rgba(0, 1, 73, 0.2);
    border-radius: var(--radius-sm);
    background: #0f172a;
    color: #f8fafc;
    padding: calc(var(--spacing-lg) + 1.4rem) var(--spacing-md) var(--spacing-md);
    width: 100%;
    max-width: 100%;
  }

  .sparql-block__label {
    position: absolute;
    top: var(--spacing-sm);
    left: var(--spacing-sm);
    background: rgba(15, 17, 26, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: var(--radius-sm);
    padding: 0.2rem 0.6rem;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  .sparql-block__code {
    overflow-x: auto;
    max-width: 100%;
  }

  pre {
    margin: 0;
    min-width: 100%;
    max-width: 100%;
  }

  code {
    display: inline-block;
    min-width: 100%;
    font-family: 'Fira Code', 'SFMono-Regular', Consolas, 'Liberation Mono',
      monospace;
    font-size: 0.85rem;
    white-space: pre;
  }

  .item {
    cursor: pointer;
    border-radius: 3px;
    border-bottom: 2px solid transparent;
  }

  .item:hover,
  .item:focus-visible {
    outline: 1px solid rgba(255, 255, 255, 0.6);
  }

  .item--retrieved {
    background: rgba(0, 166, 135, 0.22);
    border-bottom-color: rgb(0, 190, 155);
  }

  .item--verified {
    background: rgba(230, 200, 40, 0.2);
    border-bottom-color: rgb(230, 200, 40);
  }

  .item--instructed {
    background: rgba(110, 140, 235, 0.25);
    border-bottom-color: rgb(110, 140, 235);
  }

  .item--unverified {
    background: rgba(235, 70, 70, 0.25);
    border-bottom-color: rgb(245, 90, 90);
  }

  .sparql-block__actions {
    position: absolute;
    top: var(--spacing-sm);
    right: var(--spacing-sm);
    display: inline-flex;
    gap: var(--spacing-xs);
  }

  .sparql-link {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    border-radius: var(--radius-sm);
    padding: 0.3rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 600;
    background: rgba(52, 74, 154, 0.2);
    color: #fff;
    backdrop-filter: blur(2px);
    text-decoration: none;
    border: 1px solid rgba(52, 74, 154, 0.5);
    transition: transform 0.2s ease;
  }

  .sparql-link:hover {
    transform: translateY(-1px);
  }

  @media (max-width: 640px) {
    .sparql-block {
      padding: var(--spacing-md);
    }

    .sparql-block__label {
      position: static;
      margin-bottom: var(--spacing-xs);
    }

    .sparql-block__actions {
      position: static;
      margin-bottom: var(--spacing-xs);
      gap: var(--spacing-xs);
    }
  }
</style>
