<script>
  import MarkdownContent from './MarkdownContent.svelte';
  import BooleanResult from './BooleanResult.svelte';
  import TableResult from './TableResult.svelte';

  // execution result of a query: a table, boolean or text result, or a
  // formatted string in older outputs
  export let result;
</script>

{#if result?.type === 'table'}
  <TableResult {result} />
{:else if result?.type === 'boolean'}
  <BooleanResult {result} />
{:else if typeof result === 'string'}
  <MarkdownContent content={result} />
{:else if result?.formatted}
  <MarkdownContent content={result.formatted} />
{/if}
