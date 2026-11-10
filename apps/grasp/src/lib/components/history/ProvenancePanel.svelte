<script>
  import { tick } from 'svelte';
  import SparqlBlock from '../common/SparqlBlock.svelte';
  import ExecutionResult from './ExecutionResult.svelte';
  import KgItem from './KgItem.svelte';
  import { provenanceJump } from '../../stores.js';
  import { displayValue as display } from '../../utils/tables.js';

  export let sparql = null;
  export let qleverLink = null;
  // execution result as the model saw it, from the output
  export let result = null;
  // trace provenance from the output
  export let trace = null;

  const STATUS = {
    retrieved: { text: 'surfaced before use', title: 'A tool result showed this item before the query used it' },
    verified: { text: 'surfaced after use', title: 'The query used this item before a tool result showed it' },
    instructed: { text: 'from instruction', title: 'No tool result showed this item, but the instruction or its notes contain it' },
    unverified: { text: 'never surfaced', title: 'No tool result showed this item, it may be made up' }
  };

  // tooltip over a query item: { item, left, top, width, above }
  let sparqlWrapEl;
  let tipEl;
  let tip = null;
  let hideTimer = null;

  function clamp(value, min, max) {
    return Math.max(min, Math.min(value, max));
  }

  // ---- query items: tooltip and jumps to the history ----

  function cancelHide() {
    clearTimeout(hideTimer);
    hideTimer = null;
  }

  function scheduleHide() {
    cancelHide();
    // short delay, so the mouse can move from the item onto the tooltip
    hideTimer = setTimeout(() => (tip = null), 200);
  }

  // hovering with a mouse previews the tooltip of an item, clicking or
  // tapping pins it until a click outside or Escape
  function hoverItem(event) {
    if (tip?.pinned) return;
    openTip(event.detail.item, event.detail.target, false);
  }

  function leaveItem() {
    if (!tip?.pinned) scheduleHide();
  }

  function clickItem(event) {
    const { item, target } = event.detail;
    if (tip?.pinned && tip.item === item) {
      tip = null;
      return;
    }
    openTip(item, target, true);
  }

  function openTip(item, target, pinned) {
    if (!sparqlWrapEl || !target) return;
    cancelHide();
    const wrap = sparqlWrapEl.getBoundingClientRect();
    const rect = target.getBoundingClientRect();
    const width = Math.min(460, wrap.width);
    // above the item if there is not enough room below, e.g. because of the
    // composer at the bottom of the page
    const above = rect.bottom + 260 > window.innerHeight && rect.top > 260;
    tip = {
      item,
      pinned,
      width,
      above,
      left: clamp(rect.left - wrap.left, 0, wrap.width - width),
      // overlap the item a little, so there is no gap to cross with the mouse
      top: above ? rect.top - wrap.top + 2 : rect.bottom - wrap.top - 2
    };
  }

  async function handleSparqlKeydown(event) {
    if (!tip) return;
    if (event.key === 'Escape') {
      tip = null;
    } else if (event.key === 'Tab' && !event.shiftKey && event.target?.classList?.contains('item')) {
      // move into the tooltip's links before the next item
      const first = tipEl?.querySelector('button');
      if (first) {
        event.preventDefault();
        cancelHide();
        await tick();
        first.focus();
      }
    }
  }

  // which: "shown" for the call that first showed the item, "used" for the
  // first call that used it in its arguments
  function jumpTo(item, which) {
    const ref = which === 'shown' ? item.shown_by : item.first_used_by;
    if (!ref) return;
    provenanceJump.set({ ref, prefer: which === 'shown' ? 'result' : 'args' });
    tip = null;
  }

  function itemName(item) {
    return item.label ? `${item.label} (${item.written})` : item.written;
  }

  function callArgs(call) {
    return Object.entries(call?.args ?? {})
      .filter(([key]) => key !== 'kg')
      .map(([key, value]) => {
        let text = typeof value === 'string' ? value : JSON.stringify(value);
        text = String(text).replace(/\s+/g, ' ').trim();
        return [key, text.length > 60 ? `${text.slice(0, 60)}…` : text];
      });
  }

  // the calls that showed and first used an item, ordered by step; one call
  // if it did both
  function itemCalls(item) {
    const calls = [];
    if (item.shown_by) calls.push({ which: 'shown', text: 'Shown by', ref: item.shown_by });
    if (item.first_used_by && item.first_used_by.step !== item.shown_by?.step) {
      calls.push({ which: 'used', text: 'First used by', ref: item.first_used_by });
    }
    return calls.sort((a, b) => a.ref.step - b.ref.step);
  }

  function handleWindowClick(event) {
    const target = event.target;
    if (tip?.pinned && !tipEl?.contains(target) && !target?.closest?.('.item')) tip = null;
  }

  function handleWindowKeydown(event) {
    if (event.key === 'Escape') tip = null;
  }

  $: traceItems = trace?.items ?? [];
  $: marks = traceItems.flatMap((item) => item.spans.map(([start, end]) => ({ start, end, item })));
  $: statuses = new Set(traceItems.map((item) => item.status));
</script>

<svelte:window on:click={handleWindowClick} on:keydown={handleWindowKeydown} />

<div class="provenance">
  {#if result}
    <ExecutionResult {result} />
  {/if}

  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div class="sparql" bind:this={sparqlWrapEl} on:keydown={handleSparqlKeydown}>
    <SparqlBlock
      code={sparql}
      {qleverLink}
      label="SPARQL"
      {marks}
      on:itemhover={hoverItem}
      on:itemleave={leaveItem}
      on:itemclick={clickItem}
    />

    {#if tip}
      <div
        class="tip"
        class:tip--above={tip.above}
        bind:this={tipEl}
        role="tooltip"
        style={`left:${tip.left}px;top:${tip.top}px;width:${tip.width}px;`}
        on:mouseenter={cancelHide}
        on:mouseleave={leaveItem}
      >
        <div class="tip__head">
          <strong>{itemName(tip.item)}</strong>
          <span class={`badge badge--${tip.item.status}`}>{STATUS[tip.item.status]?.text ?? tip.item.status}</span>
        </div>

        <!-- the part of the tool result that showed the item -->
        {#if tip.item.shown_by?.shown}
          {@const shown = tip.item.shown_by.shown}
          <div class="tip__shown">
            {#if shown.type === 'item'}
              <KgItem item={shown.item} />
            {:else if shown.type === 'row'}
              <div class="cells">
                {#each shown.columns.filter(Boolean) as column}
                  {@const value = display(shown.row[column.variable])}
                  {#if value.text}
                    <span class="cell" class:cell--item={column.variable === shown.variable}>
                      <span class="cell__name">{column.name}</span>
                      {value.text}
                    </span>
                  {/if}
                {/each}
              </div>
            {:else if shown.type === 'example'}
              <p class="tip__example">{shown.question}</p>
              <code>{shown.line}</code>
            {:else}
              <code>{shown.text}</code>
            {/if}
          </div>
        {:else if tip.item.instruction}
          <p class="tip__note">{STATUS.instructed.title}:</p>
          <p class="tip__instruction">{tip.item.instruction.text}</p>
        {:else if !tip.item.shown_by}
          <p class="tip__note">{STATUS.unverified.title}.</p>
        {/if}

        {#if itemCalls(tip.item).length > 0}
          <div class="tip__calls">
            {#each itemCalls(tip.item) as call (call.which)}
              <button type="button" class="call" on:click={() => jumpTo(tip.item, call.which)}>
                <span class="call__what">{call.text} step {call.ref.step}</span>
                <span class="call__fn">{call.ref.tool}</span>
                {#each callArgs(call.ref) as [key, value]}
                  <span class="call__arg"><span class="call__key">{key}</span> {value}</span>
                {/each}
              </button>
            {/each}
          </div>
        {/if}
      </div>
    {/if}
  </div>

  {#if traceItems.length > 0}
    <p class="legend">
      {#each Object.entries(STATUS) as [status, info]}
        {#if statuses.has(status)}
          <span class="legend__item" title={info.title}>
            <span class={`swatch swatch--${status}`} aria-hidden="true"></span>{info.text}
          </span>
        {/if}
      {/each}
      <span class="legend__hint">Click an item to see where it came from.</span>
    </p>
  {/if}
</div>

<style>
  .provenance {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
  }

  .sparql {
    position: relative;
  }

  .tip {
    position: absolute;
    z-index: 20;
    background: #fff;
    border: 1px solid rgba(52, 74, 154, 0.3);
    border-radius: var(--radius-sm);
    box-shadow: 0 12px 28px rgba(15, 15, 47, 0.18);
    font-size: 0.85rem;
    box-sizing: border-box;
  }

  .tip {
    padding: 0.55rem 0.7rem;
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
  }

  .tip--above {
    transform: translateY(-100%);
  }

  .tip__head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem;
    overflow-wrap: anywhere;
    color: var(--color-uni-dark-blue);
  }

  .tip__note {
    margin: 0;
    color: var(--text-subtle);
    font-size: 0.82rem;
  }

  /* the instruction line containing the item, like a shown result part */
  .tip__instruction {
    margin: 0;
    max-height: 10rem;
    overflow-y: auto;
    padding: 0.35rem 0.5rem;
    border-left: 3px solid rgba(52, 74, 154, 0.25);
    background: rgba(52, 74, 154, 0.04);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    font-size: 0.85rem;
    line-height: 1.55;
    overflow-wrap: anywhere;
  }

  .tip__example {
    margin: 0 0 0.2rem;
  }

  .tip__shown {
    max-height: 10rem;
    overflow-y: auto;
    padding: 0.35rem 0.5rem;
    border-left: 3px solid rgba(52, 74, 154, 0.25);
    background: rgba(52, 74, 154, 0.04);
    border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
    font-size: 0.85rem;
    line-height: 1.55;
    overflow-wrap: anywhere;
  }

  .cells {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
  }

  .cell {
    padding: 0.1rem 0.5rem;
    border: 1px solid rgba(52, 74, 154, 0.18);
    border-radius: var(--radius-sm);
    background: rgba(52, 74, 154, 0.04);
  }

  .cell--item {
    border-color: rgba(52, 74, 154, 0.55);
    background: rgba(52, 74, 154, 0.1);
  }

  .cell__name {
    margin-right: 0.25rem;
    font-size: 0.75rem;
    color: var(--text-subtle);
  }

  .tip__calls {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    padding-top: 0.35rem;
    border-top: 1px solid rgba(52, 74, 154, 0.12);
  }

  .call {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.3rem;
    width: 100%;
    padding: 0.25rem 0.35rem;
    border: none;
    border-radius: var(--radius-sm);
    background: none;
    text-align: left;
    font: inherit;
    font-size: 0.75rem;
    color: var(--text-subtle);
    cursor: pointer;
  }

  .call:hover,
  .call:focus-visible {
    background: rgba(52, 74, 154, 0.08);
    outline: none;
  }

  .call__what {
    font-weight: 600;
    color: var(--color-uni-blue);
  }

  .call__what::after {
    content: ' →';
  }

  .call__fn,
  .call__arg {
    padding: 0.05rem 0.4rem;
    border-radius: 999px;
    border: 1px solid rgba(190, 170, 60, 0.45);
    background: #fff;
    overflow-wrap: anywhere;
  }

  .call__fn {
    font-weight: 600;
    color: var(--text-primary);
  }

  .call__key {
    font-weight: 700;
    color: var(--color-uni-yellow);
  }

  .badge {
    border-radius: 999px;
    padding: 0.05rem 0.5rem;
    font-size: 0.72rem;
    font-weight: 600;
  }

  .badge--retrieved {
    background: color-mix(in srgb, var(--color-uni-green) 16%, transparent);
    color: #006b57;
  }

  .badge--verified {
    background: color-mix(in srgb, var(--color-uni-yellow) 22%, transparent);
    color: #6b5d12;
  }

  .badge--instructed {
    background: color-mix(in srgb, var(--color-uni-blue) 14%, transparent);
    color: var(--color-uni-blue);
  }

  .badge--unverified {
    background: color-mix(in srgb, var(--color-uni-red) 14%, transparent);
    color: var(--color-uni-red);
  }

  .legend {
    margin: 0;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.35rem 0.9rem;
    font-size: 0.78rem;
    color: var(--text-subtle);
  }

  .legend__item {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
  }

  .swatch {
    width: 0.8rem;
    height: 0.8rem;
    border-radius: 3px;
  }

  .swatch--retrieved {
    background: rgb(0, 190, 155);
  }

  .swatch--verified {
    background: rgb(230, 200, 40);
  }

  .swatch--instructed {
    background: rgb(110, 140, 235);
  }

  .swatch--unverified {
    background: rgb(245, 90, 90);
  }
</style>
