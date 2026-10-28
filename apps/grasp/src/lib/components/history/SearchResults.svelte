<script>
  import KgItem from './KgItem.svelte';

  export let result;

  $: items = Array.isArray(result?.items) ? result.items : [];
  $: pages = `page ${result?.page} of ${result?.total_pages}${result?.more ? '+' : ''}`;
  // older results name the type of their items, e.g. literal alternatives
  $: label = result?.label || (items.length ? 'Results' : 'No results');
</script>

<div class="search-results">
  {#if result?.note}
    <p class="note">{result.note.trim()}</p>
  {/if}

  <p class="list-label">
    {label} <span class="list-label__meta">· {pages}</span>
  </p>
  {#if items.length}
    <ol class="list-rows">
      {#each items as item, index}
        <li class="list-row">
          <span class="list-row__rank">{(result.start ?? 0) + index + 1}</span>
          <div class="list-row__body"><KgItem {item} path={`items.${index}`} /></div>
        </li>
      {/each}
    </ol>
  {/if}
</div>

<style>
  .search-results {
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--text-primary);
    overflow-wrap: anywhere;
  }

  .note {
    margin: 0 0 var(--spacing-sm);
    white-space: pre-wrap;
    color: var(--text-subtle);
  }
</style>
