<script>
  export let item;

  $: href = /^https?:\/\//.test(item.identifier) ? item.identifier : null;
  $: identifier = item.short_identifier || item.identifier;
  $: showMatchedLabel = item.matched_label && item.matched_label !== item.label;

  function aliasText(item) {
    const aliases = item.aliases.map((alias) => `"${alias}"`).join(', ');
    const etc = item.num_aliases > item.aliases.length ? ', etc.' : '';
    return `${item.label ? 'also known as' : 'known as'} ${aliases}${etc}`;
  }
</script>

{#if item.label}
  <strong>{item.label}</strong>
{/if}
{#if href}
  <a {href} target="_blank" rel="noopener noreferrer"><code>{identifier}</code></a>
{:else}
  <code>{identifier}</code>
{/if}
{#if item.variants?.length}
  <span class="subtle">as {item.variants.join('/')}</span>
{/if}
{#if showMatchedLabel}
  <span class="subtle">matched via "{item.matched_label}"</span>
{/if}
{#if item.aliases?.length}
  <div class="subtle">{aliasText(item)}</div>
{/if}
{#if item.infos?.length}
  <ul>
    {#each item.infos as info}
      <li>{info}</li>
    {/each}
  </ul>
{/if}

<style>
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
