<script>
  import SearchItem from './SearchItem.svelte';

  export let result;

  $: items = Array.isArray(result?.items) ? result.items : [];
  $: suffix = `page ${result?.page} of ${result?.total_pages}${result?.more ? '+' : ''}`;
</script>

<div class="search-results">
  {#if result?.note}
    <p class="note">{result.note.trim()}</p>
  {/if}

  {#if items.length}
    <p>Results ({suffix}):</p>
    <ol start={(result.start ?? 0) + 1}>
      {#each items as item}
        <li><SearchItem {item} /></li>
      {/each}
    </ol>
  {:else}
    <p>No results ({suffix})</p>
  {/if}
</div>

<style>
  .search-results {
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--text-primary);
    overflow-wrap: anywhere;
  }

  .search-results p {
    margin: 0 0 var(--spacing-xs);
  }

  .search-results ol {
    margin: 0;
    padding-left: 2.5em;
  }

  .note {
    white-space: pre-wrap;
    color: var(--text-subtle);
  }
</style>
