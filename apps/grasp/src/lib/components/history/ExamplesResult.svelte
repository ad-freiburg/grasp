<script>
  import MarkdownContent from '../common/MarkdownContent.svelte';
  import { highlightQuerySpans } from '../../utils/ranges.js';
  import SparqlBlock from '../common/SparqlBlock.svelte';
  import ExecutionResult from './ExecutionResult.svelte';
  import SelectionGroups from './SelectionGroups.svelte';

  export let result;

  $: examples = Array.isArray(result?.examples) ? result.examples : [];
  // per example, the element containing its query
  let exampleEls = [];
  // e.g. "Examples (page 1 of 10+)" as label and page info
  $: summary = (result?.summary ?? '').match(/^(.*?)(?: \((.*)\))?$/);
</script>

<div class="examples-result">
  {#if result?.summary}
    <p class="list-label">
      {summary?.[1] ?? result.summary}
      {#if summary?.[2]}<span class="list-label__meta">· {summary[2]}</span>{/if}
    </p>
  {/if}
  <ol class="list-rows">
    {#each examples as example, index}
      <li class="list-row">
        <span class="list-row__rank">{(result.start ?? 0) + index + 1}</span>
        <div class="list-row__body example" bind:this={exampleEls[index]}>
          <p class="question">{example.question}</p>
          <!-- result above the query, like for the final answer -->
          <ExecutionResult result={example.result} path={`examples.${index}.result`} />
          <SparqlBlock code={example.sparql} label="SPARQL" />
          <!-- items of the query per type, older results have them formatted -->
          {#if Array.isArray(example.selections)}
            <SelectionGroups
              groups={example.selections}
              path={`examples.${index}.selections`}
              on:itemhover={(event) => highlightQuerySpans(exampleEls[index], event.detail.spans)}
            />
          {:else if example.selections}
            <MarkdownContent content={example.selections} />
          {/if}
        </div>
      </li>
    {/each}
  </ol>
</div>

<style>
  .examples-result {
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--text-primary);
  }

  .example {
    display: grid;
    /* does not grow with wide content like queries or tables */
    grid-template-columns: minmax(0, 1fr);
    gap: var(--spacing-sm);
    padding-bottom: var(--spacing-sm);
  }

  .question {
    margin: 0;
    font-weight: 600;
  }
</style>
