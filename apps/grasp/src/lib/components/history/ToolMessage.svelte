<script>
  import { getContext, tick } from 'svelte';
  import MessageCard from './MessageCard.svelte';
  import MarkdownContent from '../common/MarkdownContent.svelte';
  import SparqlBlock from '../common/SparqlBlock.svelte';
  import AnnotationsResult from './AnnotationsResult.svelte';
  import ExamplesResult from './ExamplesResult.svelte';
  import SearchResults from './SearchResults.svelte';
  import ShapesResult from './ShapesResult.svelte';
  import BooleanResult from './BooleanResult.svelte';
  import TableResult from './TableResult.svelte';
  import ProvenancePanel from './ProvenancePanel.svelte';
  import CallRow from './CallRow.svelte';
  import ItemHead from './ItemHead.svelte';
  import { provenanceJump } from '../../stores.js';
  import { STATUS } from '../../utils/provenance.js';
  import {
    flattenFunctionArgs,
    nestBulletsUnderEnumeration
  } from '../../utils/formatters.js';

  export let message;

  // old outputs and shares have plain string results
  const result =
    typeof message?.result === 'string'
      ? { type: 'text', formatted: message.result }
      : message?.result ?? null;

  const flattened = flattenFunctionArgs(message?.args ?? {});
  const argChips = [];
  let sparql = null;

  for (const entry of flattened) {
    if (entry.key === 'sparql') {
      sparql = entry.value;
      continue;
    }
    const formattedValue = coerceArgValue(entry.value);
    argChips.push({
      key: entry.key,
      value: formattedValue,
      truncated: formattedValue.length > 128
    });
  }

  const qleverLink = null;

  // where the items in the arguments come from, compared with the calls
  // before; chips are marked by the status of their first item
  const provenance = message?.provenance ?? null;
  const sparqlItems = (provenance?.items ?? []).some((item) => item.spans.length > 0);
  const chipItems = new Map();
  for (const item of provenance?.items ?? []) {
    for (const key of item.args ?? []) {
      if (!chipItems.has(key)) chipItems.set(key, item);
    }
  }

  function chipTitle(chip) {
    const item = chipItems.get(chip.key);
    const title = `${chip.key}: ${chip.value}`;
    if (!item) return title;
    const status = STATUS[item.status]?.title ?? item.status;
    return `${title}\n${status}${item.shown_by ? `, click to see step ${item.shown_by.step}` : ''}`;
  }

  function jumpToShown(chip) {
    const ref = chipItems.get(chip.key)?.shown_by;
    if (ref) provenanceJump.set({ ref, prefer: 'result' });
  }

  // the items this call showed first that later calls or answers use; the
  // result marks where it shows them, a chip lists those it shows nowhere
  const laterUses = getContext('laterUses');
  $: later = [...($laterUses?.get(message?.id)?.values() ?? [])];
  let resultEl;
  let wrapEl;
  let unmarked = [];
  let tip = null;
  // entries of the marked parts, read when opening a tip
  const laterParts = new WeakMap();
  $: if (resultEl) markLater(later);

  async function markLater(later) {
    await tick();
    const marked = new Map();
    unmarked = [];
    for (const entry of later) {
      const parts = (entry.item.shown_by?.result_paths ?? []).flatMap((path) => [
        ...resultEl.querySelectorAll(`[data-path="${CSS.escape(path)}"]`)
      ]);
      if (parts.length === 0) unmarked.push(entry);
      for (const part of parts) {
        if (!marked.has(part)) marked.set(part, []);
        marked.get(part).push(entry);
      }
    }
    for (const part of resultEl.querySelectorAll('[data-later]')) {
      if (marked.has(part)) continue;
      delete part.dataset.later;
      part.removeAttribute('tabindex');
    }
    for (const [part, entries] of marked) {
      const uses = entries.flatMap((entry) => entry.uses);
      part.dataset.later = [...new Set(uses.map((use) => use.short))].join(', ');
      part.tabIndex = 0;
      laterParts.set(part, entries);
    }
  }

  function openTip(target, entries) {
    const wrap = wrapEl.getBoundingClientRect();
    const rect = target.getBoundingClientRect();
    const width = Math.min(360, wrap.width);
    tip = {
      entries,
      target,
      width,
      left: Math.max(0, Math.min(rect.left - wrap.left, wrap.width - width)),
      top: rect.bottom - wrap.top + 4
    };
  }

  // clicks on links in a marked part follow the link
  function handleResultEvent(event) {
    if (event.type === 'keydown' && event.key !== 'Enter') return;
    if (event.target.closest('a')) return;
    const part = event.target.closest('[data-later]');
    if (!part || !resultEl.contains(part)) return;
    event.preventDefault();
    if (tip?.target === part) tip = null;
    else openTip(part, laterParts.get(part) ?? []);
  }

  function handleWindowClick(event) {
    if (!tip || wrapEl?.contains(event.target) || tip.target.contains(event.target)) return;
    tip = null;
  }

  function handleWindowKeydown(event) {
    if (tip && event.key === 'Escape') tip = null;
  }

  function jumpToUse(use) {
    tip = null;
    provenanceJump.set(
      use.ref ? { ref: use.ref, prefer: 'args' } : { answer: use.answer, spans: use.spans }
    );
  }

  // step and time limits after this call, only set with show_limits
  const limits = message?.limits ?? null;
  const limitParts = limits
    ? [
        `Step ${limits.steps} of ${limits.max_steps}`,
        ...(typeof limits.max_time === 'number'
          ? [`${Math.round(limits.elapsed)} s of ${Math.round(limits.max_time)} s`]
          : [])
      ]
    : [];
  const limitReached =
    limits !== null &&
    (limits.steps >= limits.max_steps ||
      (typeof limits.max_time === 'number' && limits.elapsed >= limits.max_time));

  function coerceArgValue(value) {
    if (value === null || value === undefined) {
      return '';
    }
    if (typeof value === 'string') {
      return normalizeWhitespace(value);
    }
    if (typeof value === 'number' || typeof value === 'boolean') {
      return String(value);
    }
    try {
      return normalizeWhitespace(JSON.stringify(value));
    } catch (error) {
      console.warn('Failed to stringify function argument', error);
      return normalizeWhitespace(String(value));
    }
  }

  function normalizeWhitespace(text) {
    return text.replace(/\s+/g, ' ').trim();
  }
</script>

<svelte:window on:click={handleWindowClick} on:keydown={handleWindowKeydown} />

<MessageCard title="Function Call" accent="var(--color-uni-yellow)">
  <svelte:fragment slot="meta">
    <div class="meta-group">
      {#if message?.name}
        <span class="function-chip">
          <span class="function-chip__key">function</span>
          <span class="function-chip__value">{message.name}</span>
        </span>
      {/if}
      {#each argChips as chip (chip.key)}
        {@const item = chipItems.get(chip.key)}
        {@const chipClass = `arg-chip${item ? ` arg-chip--${item.status}` : ''}`}
        <!-- chips of items a call before showed jump to that call -->
        {#if item?.shown_by}
          <button
            type="button"
            class={chipClass}
            title={chipTitle(chip)}
            data-arg={chip.key}
            on:click={() => jumpToShown(chip)}
          >
            <span class="arg-chip__key">{chip.key}</span>
            <span
              class="arg-chip__value"
              class:arg-chip__value--truncated={chip.truncated}
              data-full={chip.value}
            >
              {chip.value}
            </span>
          </button>
        {:else}
          <span class={chipClass} title={chipTitle(chip)} data-arg={chip.key}>
            <span class="arg-chip__key">{chip.key}</span>
            <span
              class="arg-chip__value"
              class:arg-chip__value--truncated={chip.truncated}
              data-full={chip.value}
            >
              {chip.value}
            </span>
          </span>
        {/if}
      {/each}
      {#if unmarked.length}
        <button
          type="button"
          class="later-chip"
          title="Items this call showed first that later calls or answers use"
          on:click={(event) => {
            if (tip?.target === event.currentTarget) tip = null;
            else openTip(event.currentTarget, unmarked);
          }}
        >
          {unmarked.length} used later
        </button>
      {/if}
    </div>

  </svelte:fragment>

  <!-- data attributes mark where provenance highlights items -->
  {#if sparql}
    <div data-arg="sparql">
      {#if sparqlItems}
        <ProvenancePanel {sparql} {qleverLink} trace={provenance} legend={false} />
      {:else}
        <SparqlBlock code={sparql} qleverLink={qleverLink} label="SPARQL" />
      {/if}
    </div>
  {/if}

  <!-- data-later on result parts marks the items later calls use -->
  <div class="later-wrap" bind:this={wrapEl}>
  <!-- svelte-ignore a11y_no_static_element_interactions -->
  <div
    data-part="result"
    class="result"
    bind:this={resultEl}
    on:click={handleResultEvent}
    on:keydown={handleResultEvent}
  >
  {#if result?.type === 'search'}
    <SearchResults {result} />
  {:else if result?.type === 'boolean'}
    <BooleanResult {result} />
  {:else if result?.type === 'table'}
    <TableResult {result} />
  {:else if result?.type === 'sparql_qa_examples'}
    <ExamplesResult {result} />
  {:else if result?.type === 'shapes'}
    <ShapesResult {result} />
  {:else if result?.type === 'cea_annotations' || result?.type === 'el_annotations'}
    <AnnotationsResult {result} />
  {:else if result?.formatted}
    <MarkdownContent content={nestBulletsUnderEnumeration(result.formatted)} />
  {/if}
  </div>

  {#if tip}
    <div
      class="tip"
      role="dialog"
      style={`left:${tip.left}px;top:${tip.top}px;width:${tip.width}px;`}
    >
      {#each tip.entries as { item, uses } (item.identifier)}
        <ItemHead {item} />
        <div class="tip__calls">
          {#each uses as use}
            <CallRow
              what={`Used by ${use.text}`}
              tool={use.tool}
              args={use.args}
              on:click={() => jumpToUse(use)}
            />
          {/each}
        </div>
      {/each}
    </div>
  {/if}
  </div>

  {#if limitParts.length}
    <footer class="limits" class:limits--reached={limitReached}>
      {limitParts.join(' · ')}{limitReached ? ' · limit reached' : ''}
    </footer>
  {/if}
</MessageCard>

<style>
  .function-chip {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-xs);
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    background: #fff;
    border: 1px solid rgba(190, 170, 60, 0.6);
    font-size: 0.75rem;
    white-space: nowrap;
  }

  .function-chip__key {
    font-weight: 700;
    color: var(--color-uni-yellow);
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }

  .function-chip__value {
    color: var(--text-primary);
  }

  .arg-chip {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-xs);
    padding: 0.25rem 0.65rem;
    border-radius: var(--radius-sm);
    background: #fff;
    border: 1px solid rgba(190, 170, 60, 0.6);
    font-size: 0.75rem;
    max-width: clamp(240px, 40vw, 560px);
    position: relative;
  }

  button.arg-chip {
    font: inherit;
    font-size: 0.75rem;
    color: inherit;
    text-align: left;
    cursor: pointer;
  }

  /* chips with an item, colored like the items of a query */
  .arg-chip--retrieved {
    border-color: rgb(0, 190, 155);
    background: rgba(0, 166, 135, 0.08);
  }

  .arg-chip--verified {
    border-color: rgb(230, 200, 40);
    background: rgba(230, 200, 40, 0.1);
  }

  .arg-chip--noted {
    border-color: rgb(110, 140, 235);
    background: rgba(110, 140, 235, 0.1);
  }

  .arg-chip--unverified {
    border-color: rgb(245, 90, 90);
    background: rgba(235, 70, 70, 0.08);
  }

  .arg-chip__key {
    font-weight: 700;
    color: var(--color-uni-yellow);
  }

  .arg-chip__value {
    color: var(--text-primary);
    display: inline-block;
    max-width: 128ch;
    white-space: nowrap;
    overflow: hidden;
    position: relative;
  }

  .arg-chip__value::after {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    bottom: 0;
    width: 3ch;
    pointer-events: none;
    background: linear-gradient(90deg, rgba(255, 255, 255, 0), rgba(255, 255, 255, 1));
    display: none;
  }

  .arg-chip__value--truncated::after {
    display: block;
  }

  .later-chip {
    padding: 0.25rem 0.65rem;
    border-radius: var(--radius-sm);
    border: 1px solid rgba(52, 74, 154, 0.35);
    background: rgba(52, 74, 154, 0.06);
    font: inherit;
    font-size: 0.75rem;
    color: var(--color-uni-blue);
    cursor: pointer;
  }

  .later-wrap {
    position: relative;
  }

  .result :global([data-later]) {
    cursor: pointer;
    border-radius: 3px;
    background: rgba(52, 74, 154, 0.07);
  }

  .result :global([data-later]:hover),
  .result :global([data-later]:focus-visible) {
    background: rgba(52, 74, 154, 0.14);
    outline: none;
  }

  /* the steps and answers using the item */
  .result :global([data-later]::after) {
    content: '→ ' attr(data-later);
    margin-left: 0.35rem;
    padding: 0 0.3rem;
    border-radius: 999px;
    background: rgba(52, 74, 154, 0.12);
    color: var(--color-uni-blue);
    font-size: 0.7rem;
    font-weight: 600;
    white-space: nowrap;
  }

  .tip {
    position: absolute;
    z-index: 20;
    box-sizing: border-box;
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
    padding: 0.55rem 0.7rem;
    background: #fff;
    border: 1px solid rgba(52, 74, 154, 0.3);
    border-radius: var(--radius-sm);
    box-shadow: 0 12px 28px rgba(15, 15, 47, 0.18);
    font-size: 0.85rem;
  }

  .tip__calls {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    padding-top: 0.35rem;
    border-top: 1px solid rgba(52, 74, 154, 0.12);
  }

  .limits {
    margin-top: var(--spacing-sm);
    font-size: 0.75rem;
    color: var(--text-subtle);
    text-align: right;
  }

  .limits--reached {
    color: var(--color-uni-red);
    font-weight: 600;
  }

</style>
