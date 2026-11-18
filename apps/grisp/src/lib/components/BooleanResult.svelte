<script>
  export let result;
</script>

<div class="boolean-result">
  {#if result?.summary}
    <p class="result-summary">{result.summary}</p>
  {/if}
  <span class="value" class:value--true={result?.value} class:value--false={!result?.value}>
    <span class="icon" aria-hidden="true">{result?.value ? '✓' : '✗'}</span>
    {result?.value ? 'true' : 'false'}
  </span>
</div>

<style>
  .boolean-result {
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--text-primary);
  }

  .boolean-result p {
    margin: 0 0 var(--spacing-xs);
  }

  .value {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.2rem 0.75rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.9rem;
    border: 1px solid transparent;
  }

  .value--true {
    color: #1d6b3a;
    background: rgba(46, 160, 90, 0.12);
    border-color: rgba(46, 160, 90, 0.35);
  }

  .value--false {
    color: #9b2c2c;
    background: rgba(200, 60, 60, 0.1);
    border-color: rgba(200, 60, 60, 0.3);
  }
</style>
