<script>
  import MarkedSparql from './MarkedSparql.svelte';
  import { parseMarked } from '../trace.js';

  export let step;

  // one short tag per raw beam, pointing at the final skeleton it ended up in
  function beamStatus(beam) {
    const target = beam.into !== null && beam.into !== undefined ? `#${beam.into + 1}` : '';
    switch (beam.status) {
      case 'kept': return `kept as ${target}`;
      case 'merged': return `merged into ${target}`;
      case 'duplicate': return `duplicate of ${target}`;
      case 'dropped': return 'dropped (beyond top-k)';
      case 'parse_failed': return 'parse failed';
      default: return beam.status;
    }
  }

  function mergedFrom(idx) {
    const sources = (step.beams ?? [])
      .map((beam, i) => ({ beam, i }))
      .filter(({ beam }) => beam.into === idx && (beam.status === 'kept' || beam.status === 'merged'))
      .map(({ i }) => i + 1);
    return sources.length > 1 ? `merged from beams ${sources.join(', ')}` : `from beam ${sources[0] ?? '?'}`;
  }
</script>

<div class="step-card">
  {#if step.type === 'skeletons'}
    <div class="step-header">
      <span class="step-badge step-badge--skeleton">Skeletons</span>
      <span class="step-count">{step.skeletons?.length ?? 0} generated</span>
    </div>
    {#if step.beams?.length}
      <div class="beams-header">Beams ({step.beams.length})</div>
      <div class="skeleton-list">
        {#each step.beams as beam, i (i)}
          <div class="beam-item">
            <div class="beam-meta">
              <span class="beam-index">Beam {i + 1}</span>
              <span class="beam-status beam-status--{beam.status}">{beamStatus(beam)}</span>
            </div>
            <pre class="skeleton-item" class:skeleton-item--faded={beam.status !== 'kept'}>{beam.skeleton}</pre>
          </div>
        {/each}
      </div>
      <div class="beams-header">Final skeletons ({step.skeletons?.length ?? 0})</div>
    {/if}
    <div class="skeleton-list">
      {#each step.skeletons ?? [] as entry, i (i)}
        <div class="beam-item">
          {#if step.beams?.length}
            <div class="beam-meta">
              <span class="beam-index">#{i + 1}</span>
              <span class="step-count">{mergedFrom(i)}</span>
            </div>
          {/if}
          <MarkedSparql segments={parseMarked(entry.skeleton, entry.order)} allWordings />
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .step-card {
    padding: var(--spacing-md) var(--spacing-lg);
    background: var(--surface-base);
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
  }

  .step-header {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
  }

  .step-badge {
    display: inline-flex;
    padding: 0.15rem 0.5rem;
    border-radius: var(--radius-sm);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    background: rgba(0, 0, 0, 0.06);
    color: var(--text-subtle);
  }

  .step-badge--skeleton {
    background: rgba(52, 74, 154, 0.1);
    color: var(--color-uni-blue);
  }

  .step-badge--selection {
    background: rgba(16, 185, 129, 0.1);
    color: #059669;
  }

  .step-count {
    font-size: 0.8rem;
    color: var(--text-subtle);
  }

  .step-meta {
    font-size: 0.8rem;
    color: var(--text-subtle);
  }

  .selection-timing {
    margin-left: auto;
    font-size: 0.72rem;
    color: var(--text-subtle);
    white-space: nowrap;
  }

  .beams-header {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-subtle);
    margin: var(--spacing-sm) 0 var(--spacing-xs);
  }

  .beam-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .beam-meta {
    display: flex;
    gap: var(--spacing-sm);
    align-items: center;
    font-size: 0.75rem;
  }

  .beam-index {
    font-weight: 600;
    color: var(--text-subtle);
  }

  .beam-status {
    padding: 0 0.4rem;
    border-radius: var(--radius-sm);
    background: rgba(100, 116, 139, 0.12);
    color: #475569;
  }

  .beam-status--kept {
    background: rgba(16, 185, 129, 0.1);
    color: #059669;
  }

  .beam-status--merged {
    background: rgba(52, 74, 154, 0.1);
    color: var(--color-uni-blue);
  }

  .beam-status--parse_failed {
    background: rgba(220, 38, 38, 0.1);
    color: #b91c1c;
  }

  .skeleton-item--faded {
    opacity: 0.6;
  }

  .skeleton-list {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-xs);
  }

  .skeleton-item {
    margin: 0;
    padding: var(--spacing-sm) var(--spacing-md);
    background: #0f172a;
    color: #f8fafc;
    border-radius: var(--radius-sm);
    font-family: 'Fira Code', 'SFMono-Regular', Consolas, monospace;
    font-size: 0.78rem;
    white-space: pre-wrap;
    word-break: break-word;
    overflow-x: auto;
  }

  .step-detail {
    margin: 0;
    font-size: 0.85rem;
    color: var(--text-primary);
  }

  .step-raw {
    margin: 0;
    padding: var(--spacing-sm);
    background: #f8fafc;
    border-radius: var(--radius-sm);
    font-size: 0.78rem;
    white-space: pre-wrap;
    word-break: break-word;
    max-height: 200px;
    overflow-y: auto;
  }

  .alternatives-section,
  .reranking-section {
    margin-top: var(--spacing-sm);
    padding: var(--spacing-sm);
    background: rgba(52, 74, 154, 0.04);
    border-radius: var(--radius-sm);
    border: 1px solid rgba(52, 74, 154, 0.15);
  }

  .alternatives-header,
  .reranking-header {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--color-uni-blue);
    margin-bottom: var(--spacing-xs);
  }

  .alternatives-list,
  .reranking-list {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-xs);
  }

  .alternative-item,
  .reranking-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: var(--spacing-xs) var(--spacing-sm);
    background: var(--surface-base);
    border-radius: var(--radius-sm);
    font-size: 0.82rem;
    gap: var(--spacing-sm);
    transition: opacity 0.2s ease, background 0.2s ease;
  }

  .alternative-item--none {
    background: rgba(52, 74, 154, 0.08);
    border: 1px solid rgba(52, 74, 154, 0.2);
    font-weight: 600;
  }

  .alternative-item--selected {
    background: rgba(0, 160, 130, 0.08);
    border: 1px solid rgba(0, 160, 130, 0.3);
    font-weight: 600;
  }

  .alternative-item--below-none {
    opacity: 0.4;
    background: rgba(0, 0, 0, 0.02);
  }

  .selected-indicator {
    color: var(--color-uni-green);
    font-weight: bold;
    margin-right: 0.3em;
  }

  .alternative-label,
  .reranking-label {
    flex: 1;
    color: var(--text-primary);
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    display: flex;
    align-items: center;
    gap: 0.3em;
    flex-wrap: wrap;
  }

  .alternative-id,
  .reranking-id {
    color: var(--text-subtle);
    font-size: 0.9em;
  }

  .original-rank {
    color: var(--text-subtle);
    font-size: 0.75rem;
    font-style: italic;
    white-space: nowrap;
  }

  .alternative-meta {
    display: flex;
    align-items: center;
    gap: var(--spacing-xs);
  }

  .alternative-score,
  .reranking-score {
    flex-shrink: 0;
    font-size: 0.75rem;
    color: var(--text-subtle);
    font-family: 'Fira Code', monospace;
  }

  .selection-events {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
  }

  .selection-event {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-xs);
    padding: var(--spacing-sm) 0;
  }

  .selection-event:not(:last-child) {
    border-bottom: 1px solid rgba(0, 0, 0, 0.06);
    padding-bottom: var(--spacing-md);
  }

  .selection-header {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
    margin-bottom: var(--spacing-xs);
  }

  .selection-chip {
    display: inline-flex;
    padding: 0.15rem 0.5rem;
    border-radius: var(--radius-sm);
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    background: rgba(52, 74, 154, 0.1);
    color: var(--color-uni-blue);
  }

  .selection-query {
    font-size: 0.85rem;
    color: var(--text-subtle);
    font-style: italic;
  }
</style>
