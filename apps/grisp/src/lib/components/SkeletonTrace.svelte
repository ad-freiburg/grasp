<script>
  import MarkedSparql from './MarkedSparql.svelte';
  import AttemptCard from './AttemptCard.svelte';
  import { circled } from '../trace.js';

  export let skeleton;
  export let isAnswer = false;

  // null until toggled, then the user's choice wins
  let toggled = null;
  $: open = toggled ?? (isAnswer || skeleton.status === 'running');

  $: badge = {
    resolved: { text: isAnswer ? '✓ answer' : '✓ resolved', kind: 'ok' },
    failed: { text: `✗ ${skeleton.reason}`, kind: 'bad' },
    running: { text: 'resolving...', kind: 'run' },
    pending: { text: 'not tried', kind: 'idle' },
  }[skeleton.status];
</script>

<div class="skeleton" class:skeleton--answer={isAnswer}>
  <button type="button" class="skeleton-header" on:click={() => (toggled = !open)}>
    <span class="arrow">{open ? '▾' : '▸'}</span>
    <span class="title">Skeleton #{skeleton.idx + 1}</span>
    <span class="badge badge--{badge.kind}">{badge.text}</span>
  </button>

  {#if open}
    <div class="skeleton-body">
      <MarkedSparql segments={skeleton.segments} allWordings />
      {#if skeleton.status !== 'pending'}
        <div class="summary">
          {#each skeleton.cards as card (card.step)}
            <div class="summary-row" class:summary-row--unreached={card.attempts.length === 0}>
              <span class="summary-step" class:summary-step--done={card.selected}>{circled(card.step)}</span>
              <span class="summary-wording">{card.wordings[0]}</span>
              <span class="summary-arrow">→</span>
              {#if card.selected}
                <strong>{card.selected.label ?? card.selected.identifier}</strong>
                <span class="summary-id">{card.selected.identifier}{card.selected.variant ? ` as ${card.selected.variant}` : ''}</span>
              {:else}
                <em>{card.attempts.length === 0 ? 'not reached' : 'unresolved'}</em>
              {/if}
            </div>
          {/each}
        </div>
        <div class="timeline">
          {#each skeleton.timeline as entry, i (i)}
            {#if entry.kind === 'attempt'}
              <AttemptCard attempt={entry.attempt} />
            {:else}
              <div class="event event--{entry.tone}">{entry.text}</div>
            {/if}
          {/each}
        </div>
      {/if}
    </div>
  {/if}
</div>

<style>
  .skeleton {
    border: 1px solid var(--border-default);
    border-radius: var(--radius-md);
    background: var(--surface-base);
    box-shadow: var(--shadow-sm);
    overflow: hidden;
  }

  .skeleton--answer {
    border-color: rgba(0, 160, 130, 0.45);
  }

  .skeleton-header {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    width: 100%;
    padding: var(--spacing-md) var(--spacing-lg);
    border: none;
    background: rgba(52, 74, 154, 0.03);
    cursor: pointer;
    font: inherit;
    text-align: left;
  }

  .arrow {
    width: 1em;
    color: var(--text-subtle);
  }

  .title {
    font-weight: 600;
    font-size: 0.92rem;
    color: var(--color-uni-dark-blue);
  }

  .badge {
    padding: 0.1rem 0.5rem;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
  }

  .badge--ok {
    background: rgba(0, 160, 130, 0.12);
    color: #047857;
  }

  .badge--bad {
    background: rgba(193, 0, 42, 0.08);
    color: #b91c1c;
  }

  .badge--run {
    background: rgba(52, 74, 154, 0.1);
    color: var(--color-uni-blue);
  }

  .badge--idle {
    background: rgba(0, 0, 0, 0.05);
    color: var(--text-subtle);
  }

  .skeleton-body {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
    padding: var(--spacing-md) var(--spacing-lg) var(--spacing-lg);
  }

  .summary {
    display: grid;
    gap: 2px;
    font-size: 0.85rem;
  }

  .summary-row {
    display: flex;
    align-items: baseline;
    gap: var(--spacing-sm);
    flex-wrap: wrap;
  }

  .summary-row--unreached {
    opacity: 0.5;
  }

  .summary-step {
    color: var(--text-subtle);
    font-size: 1rem;
  }

  .summary-step--done {
    color: var(--color-uni-green);
  }

  .summary-wording {
    color: var(--text-subtle);
  }

  .summary-arrow {
    color: var(--text-subtle);
  }

  .summary-id {
    font-family: 'Fira Code', monospace;
    font-size: 0.75rem;
    color: var(--text-subtle);
  }

  .timeline {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
    padding-top: var(--spacing-sm);
    border-top: 1px solid var(--border-default);
  }

  .event {
    padding: var(--spacing-xs) var(--spacing-md);
    border-radius: var(--radius-sm);
    font-size: 0.85rem;
    font-weight: 600;
    text-align: center;
  }

  .event--ok {
    background: rgba(0, 160, 130, 0.1);
    color: #047857;
  }

  .event--warn {
    background: rgba(190, 170, 60, 0.15);
    color: #7a6a12;
  }

  .event--bad {
    background: rgba(193, 0, 42, 0.08);
    color: #b91c1c;
  }
</style>
