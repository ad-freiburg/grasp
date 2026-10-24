<script>
  export let title = '';
  export let accent = 'var(--color-uni-blue)';
  export let subtitle = '';
</script>

<article class="message-card" style={`--accent:${accent}`}>
  <header class="message-card__header">
    <div class="message-card__title">
      <span class="message-card__marker" aria-hidden="true"></span>
      <h3>{title}</h3>
    </div>
    <div class="message-card__meta">
      <slot name="meta" />
    </div>
  </header>
  {#if subtitle}
    <p class="message-card__subtitle">{subtitle}</p>
  {/if}
  <div class="message-card__body">
    <slot />
  </div>
</article>

<style>
  .message-card {
    border-radius: var(--radius-md);
    border: 1px solid rgba(0, 0, 0, 0.05);
    background: rgba(255, 255, 255, 0.75);
    box-shadow: 0 10px 18px rgba(15, 15, 47, 0.04);
    padding: var(--spacing-md);
    display: grid;
    /* a single column that does not grow with wide content, so tables
       scroll inside it */
    grid-template-columns: minmax(0, 1fr);
    width: 100%;
    max-width: 100%;
  }

  .message-card__header {
    display: flex;
    justify-content: space-between;
    gap: var(--spacing-md);
    flex-wrap: nowrap;
    align-items: center;
    margin-bottom: var(--spacing-xs);
  }

  .message-card__title {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-sm);
    flex: 0 0 auto;
    white-space: nowrap;
  }

  .message-card__marker {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--accent) 18%, transparent);
  }

  .message-card__title h3 {
    margin: 0;
    font-size: 0.95rem;
    font-weight: 650;
    color: var(--accent);
  }

  .message-card__meta {
    margin-left: auto;
    display: inline-flex;
    flex-wrap: wrap;
    gap: var(--spacing-sm);
    align-items: center;
    justify-content: flex-end;
    /* shrink below the one-line width of all chips, otherwise they never wrap */
    flex: 0 1 auto;
    min-width: 0;
    white-space: nowrap;
  }

  .message-card__meta :global(.meta-group) {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-xs);
    flex-wrap: wrap;
  }

  .message-card__subtitle {
    margin: 0 0 var(--spacing-xs);
    color: var(--text-subtle);
    font-size: 0.85rem;
  }

  .message-card__body {
    display: grid;
    grid-template-columns: minmax(0, 1fr);
    gap: var(--spacing-sm);
  }

  .message-card__body > :global(*) {
    min-width: 0;
    max-width: 100%;
  }

  @media (max-width: 720px) {
    .message-card__header {
      flex-wrap: wrap;
    }

    .message-card__meta {
      flex-shrink: 1;
      width: 100%;
      justify-content: flex-start;
    }

    .message-card__meta :global(.meta-group) {
      flex-wrap: wrap;
      gap: var(--spacing-xs);
    }
  }
</style>
