<script>
  import EntityLinkingAnnotations from './EntityLinkingAnnotations.svelte';
  import SearchItem from './SearchItem.svelte';

  export let result;

  const NIL = '<NIL>';

  $: entities = Array.isArray(result?.entities) ? result.entities : [];

  function cellHref(cell) {
    return /^https?:\/\//.test(cell?.identifier ?? '') ? cell.identifier : null;
  }
</script>

<div class="annotations-result">
  {#if result?.message}
    <p>{result.message}</p>
  {/if}

  {#if result?.type === 'cea_annotations'}
    <table>
      <thead>
        <tr>
          <th>Row</th>
          {#each result.header as name, index}
            <th>Column {index}: {name}</th>
          {/each}
        </tr>
      </thead>
      <tbody>
        {#each result.cells as row, index}
          <tr>
            <td>{result.rows[index]}</td>
            {#each row as cell}
              <td>
                {cell.value}
                {#if cell.entity === NIL}
                  <span class="subtle">({NIL})</span>
                {:else if cell.entity && cellHref(cell)}
                  <a href={cellHref(cell)} target="_blank" rel="noopener noreferrer"
                    >({cell.entity})</a
                  >
                {:else if cell.entity}
                  <span class="subtle">({cell.entity})</span>
                {/if}
              </td>
            {/each}
          </tr>
        {/each}
      </tbody>
    </table>
  {:else if result?.type === 'el_annotations'}
    <EntityLinkingAnnotations text={result.text} predictions={result.spans} />
  {/if}

  {#if entities.length}
    <p>Annotated entities:</p>
    <ul>
      {#each entities as item}
        <li><SearchItem {item} /></li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .annotations-result {
    display: grid;
    gap: var(--spacing-sm);
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--text-primary);
    overflow-wrap: anywhere;
  }

  .annotations-result p {
    margin: 0;
  }

  table {
    border-collapse: collapse;
    display: block;
    max-width: 100%;
    overflow-x: auto;
  }

  th,
  td {
    border: 1px solid rgba(0, 0, 0, 0.08);
    padding: 6px 8px;
    text-align: left;
    white-space: nowrap;
  }

  ul {
    margin: 0;
    padding-left: 1.25em;
  }

  .subtle {
    color: var(--text-subtle);
  }

  a {
    color: var(--color-uni-blue);
    text-decoration: underline;
    text-decoration-color: rgba(52, 74, 154, 0.4);
  }
</style>
