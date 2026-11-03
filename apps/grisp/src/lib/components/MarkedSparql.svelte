<script>
  import { circled } from '../trace.js';

  // segments from parseMarked
  export let segments = [];
  // show all wordings of a placeholder, not just the first
  export let allWordings = false;
</script>

<pre class="marked">{#each segments as seg, i (i)}{#if seg.kind === 'text'}{seg.text}{:else}<span
        class="ph"
        class:ph--current={seg.kind === 'current'}
        title={seg.kind === 'current' ? 'placeholder being resolved' : 'unresolved placeholder'}
      >{#if seg.step !== null}<span class="ph-step">{circled(seg.step)}</span>{/if}{#if allWordings}{#each seg.wordings as w, j (j)}{#if j > 0}<span class="ph-sep">|</span>{/if}{w}{/each}{:else}{seg.wordings[0]}{/if}</span
      >{/if}{/each}</pre>

<style>
  .marked {
    margin: 0;
    padding: var(--spacing-sm) var(--spacing-md);
    background: #0f172a;
    color: #e2e8f0;
    border-radius: var(--radius-sm);
    font-family: 'Fira Code', 'SFMono-Regular', Consolas, monospace;
    font-size: 0.78rem;
    line-height: 1.9;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .ph {
    padding: 0.1rem 0.35rem;
    border-radius: 4px;
    border: 1px dashed rgba(148, 163, 184, 0.6);
    color: #cbd5e1;
    background: rgba(148, 163, 184, 0.12);
  }

  .ph--current {
    border: 1px solid #fbbf24;
    background: rgba(251, 191, 36, 0.2);
    color: #fde68a;
    font-weight: 600;
  }

  .ph-step {
    margin-right: 0.3em;
    font-weight: 700;
  }

  .ph-sep {
    margin: 0 0.35em;
    opacity: 0.5;
  }
</style>
