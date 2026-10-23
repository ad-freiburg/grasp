<script>
  import { displayValue } from '../utils/tables.js';

  export let result;

  $: columns = Array.isArray(result?.columns) ? result.columns : [];
  $: rows = Array.isArray(result?.rows) ? result.rows : [];
</script>

{#if result}
  <div class="table-result">
    {#if result.summary}
      <p class="result-summary">{result.summary}</p>
    {/if}
    <div class="table">
      <table>
        <thead>
          <tr>
            {#each columns as column}
              <th scope="col" class:gap={!column}>{column ? column.name : '…'}</th>
            {/each}
          </tr>
        </thead>
        <tbody>
          {#each rows as row}
            {#if row}
              <tr>
                {#each columns as column}
                  {#if column}
                    {@const value = displayValue(row[column.variable])}
                    <td>
                      {#if value.href}
                        <a href={value.href} target="_blank" rel="noopener noreferrer">{value.text}</a>
                      {:else}
                        {value.text}
                      {/if}
                      {#if value.tag}<span class="tag">{value.tag}</span>{/if}
                    </td>
                  {:else}
                    <td class="gap">…</td>
                  {/if}
                {/each}
              </tr>
            {:else}
              <tr class="gap-row">
                <td colspan={columns.length}>…</td>
              </tr>
            {/if}
          {/each}
        </tbody>
      </table>
    </div>
  </div>
{/if}

<style>
  .table-result {
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--text-primary);
  }

  .table-result p {
    margin: 0 0 var(--spacing-xs);
  }

  .table {
    overflow-x: auto;
    border: 1px solid rgba(52, 74, 154, 0.2);
    border-radius: var(--radius-sm);
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.85rem;
  }

  th,
  td {
    padding: 0.4rem 0.6rem;
    border-bottom: 1px solid rgba(52, 74, 154, 0.12);
    text-align: left;
    vertical-align: top;
  }

  tbody tr:last-child td {
    border-bottom: none;
  }

  thead {
    background: rgba(52, 74, 154, 0.08);
  }

  .gap,
  .gap-row td {
    color: var(--text-subtle);
  }

  .gap-row td {
    text-align: center;
    font-size: 0.8rem;
    padding: 0.25rem 0.6rem;
    background: rgba(52, 74, 154, 0.03);
  }

  .tag {
    margin-left: 0.3rem;
    font-size: 0.72rem;
    color: var(--text-subtle);
  }

  a {
    color: var(--color-uni-blue);
    text-decoration: none;
  }

  a:hover {
    text-decoration: underline;
  }
</style>
