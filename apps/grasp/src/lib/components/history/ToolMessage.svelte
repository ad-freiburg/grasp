<script>
  import MessageCard from './MessageCard.svelte';
  import MarkdownContent from '../common/MarkdownContent.svelte';
  import SparqlBlock from '../common/SparqlBlock.svelte';
  import AnnotationsResult from './AnnotationsResult.svelte';
  import ExamplesResult from './ExamplesResult.svelte';
  import SearchResults from './SearchResults.svelte';
  import ShapesResult from './ShapesResult.svelte';
  import TableResult from './TableResult.svelte';
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
        <span class="arg-chip" title={`${chip.key}: ${chip.value}`}>
          <span class="arg-chip__key">{chip.key}</span>
          <span
            class="arg-chip__value"
            class:arg-chip__value--truncated={chip.truncated}
            data-full={chip.value}
          >
            {chip.value}
          </span>
        </span>
      {/each}
    </div>

  </svelte:fragment>

  {#if sparql}
    <SparqlBlock code={sparql} qleverLink={qleverLink} label="SPARQL" />
  {/if}

  {#if result?.type === 'search'}
    <SearchResults {result} />
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

</style>
