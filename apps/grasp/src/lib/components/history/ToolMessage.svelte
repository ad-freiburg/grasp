<script>
  import { getContext } from 'svelte';
  import MessageCard from './MessageCard.svelte';
  import MarkdownContent from '../common/MarkdownContent.svelte';
  import SparqlBlock from '../common/SparqlBlock.svelte';
  import AnnotationsResult from './AnnotationsResult.svelte';
  import ExamplesResult from './ExamplesResult.svelte';
  import SearchResults from './SearchResults.svelte';
  import ShapesResult from './ShapesResult.svelte';
  import BooleanResult from './BooleanResult.svelte';
  import TableResult from './TableResult.svelte';
  import ProvenancePanel, { STATUS } from './ProvenancePanel.svelte';
  import { provenanceJump } from '../../stores.js';
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

  // the items this call showed first that later calls or answers use,
  // clicking a use jumps to it
  const laterUses = getContext('laterUses');
  $: later = [...($laterUses?.get(message?.id)?.values() ?? [])];

  function jumpToUse(use) {
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

  <div data-part="result">
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

  {#if later.length}
    <ul class="later">
      {#each later as { item, uses } (item.identifier)}
        <li>
          <span class="later__item" title={item.identifier}>
            {item.label ? `${item.label} (${item.written})` : item.written}
          </span>
          used later by
          {#each uses as use, index}
            <button type="button" class="later__use" on:click={() => jumpToUse(use)}>
              {use.text}</button
            >{index < uses.length - 1 ? ', ' : ''}
          {/each}
        </li>
      {/each}
    </ul>
  {/if}

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

  .later {
    list-style: none;
    margin: var(--spacing-sm) 0 0;
    padding: 0;
    font-size: 0.75rem;
    color: var(--text-subtle);
  }

  .later__item {
    color: var(--text-primary);
    font-weight: 600;
  }

  .later__use {
    padding: 0;
    border: none;
    background: none;
    font: inherit;
    color: var(--color-uni-blue);
    text-decoration: underline;
    cursor: pointer;
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
