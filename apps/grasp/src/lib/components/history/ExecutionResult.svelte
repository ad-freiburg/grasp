<script>
  import MarkdownContent from '../common/MarkdownContent.svelte';
  import BooleanResult from './BooleanResult.svelte';
  import TableResult from './TableResult.svelte';

  // execution result of a query: a table, boolean or text result, or a
  // formatted string in older outputs
  export let result;
  // path of the result in a structured result, for data-path on its values
  export let path = '';
</script>

{#if result?.type === 'table'}
  <TableResult {result} {path} />
{:else if result?.type === 'boolean'}
  <BooleanResult {result} />
{:else if typeof result === 'string'}
  <MarkdownContent content={result} />
{:else if result?.formatted}
  <MarkdownContent content={result.formatted} />
{/if}
