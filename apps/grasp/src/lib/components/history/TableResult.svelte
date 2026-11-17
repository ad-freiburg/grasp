<script>
  export let result;

  $: header = Array.isArray(result?.header) ? result.header : [];
  $: rows = Array.isArray(result?.rows) ? result.rows : [];

  // null columns, rows and values are omitted or unbound
  function display(value) {
    if (!value) return { text: '' };
    if (value.type === 'uri') {
      const short = value.short_identifier ?? value.value;
      return {
        text: value.label ? `${value.label} (${short})` : short,
        href: /^https?:\/\//.test(value.value) ? value.value : null
      };
    }
    return { text: value.value, tag: value.lang ?? null };
  }
</script>

<div class="table-result">
  {#if result?.summary}
    <p>{result.summary}:</p>
  {/if}
  <table>
    <thead>
      <tr>
        {#each header as column}
          <th>{column ?? '…'}</th>
        {/each}
      </tr>
    </thead>
    <tbody>
      {#each rows as row}
        <tr>
          {#if row}
            {#each header as column, index}
              {@const value = column == null ? null : display(row[index])}
              <td>
                {#if !value}
                  …
                {:else if value.href}
                  <a href={value.href} target="_blank" rel="noopener noreferrer">{value.text}</a>
                {:else}
                  {value.text}
                {/if}
                {#if value?.tag}<span class="tag">{value.tag}</span>{/if}
              </td>
            {/each}
          {:else}
            <td colspan={header.length}>…</td>
          {/if}
        </tr>
      {/each}
    </tbody>
  </table>
</div>

<style>
  .table-result {
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--text-primary);
  }

  .table-result p {
    margin: 0 0 var(--spacing-xs);
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

  .tag {
    margin-left: 4px;
    font-size: 0.8em;
    color: var(--text-subtle);
  }

  a {
    color: var(--color-uni-blue);
    text-decoration: underline;
    text-decoration-color: rgba(52, 74, 154, 0.4);
  }
</style>
