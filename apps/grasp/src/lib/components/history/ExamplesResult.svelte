<script>
  import MarkdownContent from '../common/MarkdownContent.svelte';
  import SparqlBlock from '../common/SparqlBlock.svelte';
  import TableResult from './TableResult.svelte';

  export let result;

  $: examples = Array.isArray(result?.examples) ? result.examples : [];
</script>

<div class="examples-result">
  {#if result?.summary}
    <p>{result.summary}:</p>
  {/if}
  {#each examples as example, index}
    <div class="example">
      <p><strong>Example {(result.start ?? 0) + index + 1}:</strong> {example.question}</p>
      <SparqlBlock code={example.sparql} label="SPARQL" />
      {#if example.selections}
        <MarkdownContent content={example.selections} />
      {/if}
      {#if example.table}
        <TableResult result={example.table} />
      {:else if example.result}
        <MarkdownContent content={example.result} />
      {/if}
    </div>
  {/each}
</div>

<style>
  .examples-result {
    display: grid;
    gap: var(--spacing-md);
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--text-primary);
  }

  .examples-result p {
    margin: 0;
  }

  .example {
    display: grid;
    gap: var(--spacing-sm);
  }
</style>
