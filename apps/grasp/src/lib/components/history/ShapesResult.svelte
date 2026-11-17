<script>
  export let result;

  $: shapes = Array.isArray(result?.shapes) ? result.shapes : [];

  function href(iri) {
    return /^https?:\/\//.test(iri ?? '') ? iri : null;
  }
</script>

<div class="shapes-result">
  {#if result?.summary}
    <p>{result.summary}:</p>
  {/if}
  <ol class:single={!result?.summary}>
    {#each shapes as shape}
      <li>
        {#if href(shape.iri)}
          <a href={href(shape.iri)} target="_blank" rel="noopener noreferrer"
            ><code>{shape.value}</code></a
          >
        {:else}
          <code>{shape.value}</code>
        {/if}
        <ul>
          {#each shape.properties as prop}
            <li>
              {#if prop.inverse}<code>^</code>{/if}{#if href(prop.iri)}<a
                  href={href(prop.iri)}
                  target="_blank"
                  rel="noopener noreferrer"><code>{prop.value}</code></a
                >{:else}<code>{prop.value}</code>{/if}
              {#if prop.targets.length}
                {#each prop.targets as target, index}
                  {#if index > 0}<span class="subtle"> | </span>{/if}
                  {#if href(target.iri)}
                    <a href={href(target.iri)} target="_blank" rel="noopener noreferrer"
                      ><code>{target.value}</code></a
                    >
                  {:else}
                    <code>{target.value}</code>
                  {/if}
                {/each}
              {:else}
                <code>IRI</code>
              {/if}
              {#if prop.cardinality}
                <span class="subtle">{prop.cardinality}</span>
              {/if}
            </li>
          {/each}
          {#if shape.note}
            <li class="subtle">{shape.note}</li>
          {/if}
        </ul>
      </li>
    {/each}
  </ol>
</div>

<style>
  .shapes-result {
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--text-primary);
    overflow-wrap: anywhere;
  }

  .shapes-result p {
    margin: 0 0 var(--spacing-xs);
  }

  ol {
    margin: 0;
    padding-left: 2.5em;
  }

  ol.single {
    list-style: none;
    padding-left: 0;
  }

  ul {
    margin: 0;
    padding-left: 1.25em;
  }

  .subtle {
    color: var(--text-subtle);
  }

  code {
    font-family: 'Fira Code', 'SFMono-Regular', Consolas, 'Liberation Mono',
      monospace;
    font-size: 0.875em;
  }

  a {
    color: var(--color-uni-blue);
    text-decoration: underline;
    text-decoration-color: rgba(52, 74, 154, 0.4);
  }
</style>
