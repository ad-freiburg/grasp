<script>
  import SparqlBlock from './SparqlBlock.svelte';
  import MarkdownContent from './MarkdownContent.svelte';
  import ExecutionResult from './ExecutionResult.svelte';
  import SelectionGroups from './SelectionGroups.svelte';
  import { highlightQuerySpans } from '../utils/ranges.js';

  export let output;

  const out = output?.output ?? {};
  const elapsed = typeof output?.elapsed === 'number' ? output.elapsed : null;
  const skeletonsTime = typeof output?.timings?.skeletons === 'number' ? output.timings.skeletons : null;
  const resolutionTime = typeof output?.timings?.resolution === 'number' ? output.timings.resolution : null;
  const error = output?.error ?? null;

  const sparql = out?.sparql ?? null;
  // items of the query per type and the execution result in structured
  // form, older outputs have them formatted
  const selections = out?.selections ?? null;
  const result = out?.result ?? null;
  // element containing the query, for highlighting hovered query items
  let queryEl;
  const formatted = out?.formatted ?? null;
  const endpoint = out?.endpoint ?? null;
</script>

<div class="output-card">
  <div class="output-header">
    <h2 class="output-title">Output</h2>
    {#if elapsed !== null}
      <span class="chip">
        {elapsed.toFixed(2)}s{#if skeletonsTime !== null && resolutionTime !== null}
          &nbsp;(skeletons {skeletonsTime.toFixed(2)}s · resolution {resolutionTime.toFixed(2)}s){/if}
      </span>
    {/if}
  </div>

  {#if error}
    <div class="error-block">
      <p class="error-reason">Error: {error.reason ?? 'unknown'}</p>
      {#if error.content}
        <pre class="error-content">{error.content}</pre>
      {/if}
    </div>
  {/if}

  {#if result}
    <ExecutionResult {result} />
  {/if}

  {#if sparql}
    <div bind:this={queryEl}>
      <SparqlBlock code={sparql} {endpoint} />
    </div>
  {/if}

  {#if Array.isArray(selections) && selections.length}
    <SelectionGroups
      groups={selections}
      on:itemhover={(event) => highlightQuerySpans(queryEl, event.detail.spans)}
    />
  {:else if typeof selections === 'string' && selections}
    <MarkdownContent content={selections} />
  {/if}

  {#if !sparql && !result && !error}
    <p class="placeholder">No output generated.</p>
  {/if}
</div>

<style>
  .output-card {
    border: 1px solid rgba(52, 74, 154, 0.25);
    border-radius: var(--radius-md);
    background: var(--surface-base);
    padding: var(--spacing-lg);
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
    box-shadow: var(--shadow-sm);
  }

  .output-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .output-title {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--color-uni-dark-blue);
  }

  .chip {
    display: inline-flex;
    align-items: center;
    padding: 0.15rem 0.6rem;
    border-radius: 999px;
    background: rgba(52, 74, 154, 0.08);
    color: var(--color-uni-blue);
    font-size: 0.75rem;
    font-weight: 600;
  }

  .error-block {
    padding: var(--spacing-md);
    background: rgba(239, 68, 68, 0.06);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-radius: var(--radius-sm);
  }

  .error-reason {
    margin: 0;
    font-weight: 600;
    color: #dc2626;
    font-size: 0.9rem;
  }

  .error-content {
    margin: var(--spacing-xs) 0 0;
    font-size: 0.85rem;
    white-space: pre-wrap;
    word-break: break-word;
    color: #7f1d1d;
  }

  .placeholder {
    margin: 0;
    color: var(--text-subtle);
    font-size: 0.9rem;
    font-style: italic;
  }
</style>
