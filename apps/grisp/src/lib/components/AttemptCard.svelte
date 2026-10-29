<script>
  import MarkedSparql from './MarkedSparql.svelte';
  import { circled } from '../trace.js';

  export let attempt;

  // as many as the selection prompt shows
  const MAX_ALIASES = 5;
  const MAX_INFO = 10;

  let constraintOpen = false;
  let expanded = {};

  // aliases and infos as the selection prompt shows them
  function aliasText(alt) {
    if (!alt.aliases?.length) return '';
    const shown = alt.aliases.slice(0, MAX_ALIASES).map((a) => `"${a}"`).join(', ');
    const etc = alt.aliases.length > MAX_ALIASES ? ', etc.' : '';
    return `${alt.label ? 'also known as' : 'known as'} ${shown}${etc}`;
  }

  function infos(alt) {
    return (alt.info ?? []).slice(0, MAX_INFO);
  }

  // older traces only have the short identifier
  function href(alt) {
    return /^https?:\/\//.test(alt.iri ?? '') ? alt.iri : null;
  }

  $: termKind = attempt.alternatives.find((a) => a.kind)?.kind ?? null;
  $: badge = constraintBadge(attempt.constraint);

  function constraintBadge(c) {
    if (!c) return null;
    if (c.candidates !== null && c.candidates !== undefined) {
      return { text: `constrained · ${c.candidates.toLocaleString()} candidate${c.candidates === 1 ? '' : 's'}`, ok: true };
    }
    return { text: `unconstrained · ${shortReason(c.reason)}`, ok: false };
  }

  // the full reason is shown with the constraint query
  function shortReason(reason) {
    if (!reason) return 'unknown';
    if (reason.includes('timed out')) return 'constraint query timed out';
    if (reason.includes('maximum supported number')) return 'too many candidates';
    return reason.length > 40 ? 'constraint query failed' : reason;
  }

  // the server names variables of placeholders randomly: the selected one
  // becomes ?candidate, unresolved ones ?open1, ?open2, ...; one triple per line
  function prettyConstraint(sparql) {
    const names = new Map();
    return sparql
      .replace(/\?[0-9a-f]{32}\b/g, (v) => {
        if (!names.has(v)) names.set(v, names.size === 0 ? '?candidate' : `?open${names.size}`);
        return names.get(v);
      })
      .replace('WHERE { ', 'WHERE {\n  ')
      .replaceAll(' . ', ' .\n  ')
      .replace(/ \}( LIMIT \d+)?$/, '\n}$1');
  }

  function formatTimings(t) {
    if (!t) return '';
    const parts = [];
    if (t.constraint > 0) parts.push(`constraint ${t.constraint.toFixed(2)}s`);
    if (t.search > 0) parts.push(`search ${t.search.toFixed(2)}s`);
    if (t.rerank > 0) parts.push(`rerank ${t.rerank.toFixed(2)}s`);
    return parts.join(' · ');
  }

  function outcomeText(o) {
    switch (o.type) {
      case 'select':
        return { icon: '✓', text: `Selected ${o.label ?? o.identifier}${o.label && o.label !== o.identifier ? ` (${o.identifier})` : ''}${o.variant ? ` as ${o.variant}` : ''}`, kind: 'ok' };
      case 'continue':
        return { icon: '✗', text: `${o.label ?? o.identifier} has no variant ${o.variant}, trying the next one`, kind: 'warn' };
      case 'next_wording':
        return { icon: '→', text: 'No fitting alternative, searching with the next wording', kind: 'warn' };
      default:
        return { icon: '·', text: o.type, kind: '' };
    }
  }
</script>

<div class="attempt">
  <div class="attempt-rail">
    <span class="step" class:step--done={attempt.outcome.some((o) => o.type === 'select')}>{circled(attempt.step)}</span>
  </div>

  <div class="attempt-body">
    <div class="attempt-header">
      <span class="attempt-title"><strong>{attempt.query}</strong></span>
      {#if attempt.wordings.length > 1}
        <span class="wording-of">wording {attempt.wording + 1} of {attempt.wordings.length}</span>
      {/if}
      {#if termKind}
        <span class="type">{termKind}</span>
      {/if}
      {#if attempt.changed}
        <span class="revisit">again, now with {attempt.changed}</span>
      {/if}
      {#if formatTimings(attempt.timings)}
        <span class="timing">{formatTimings(attempt.timings)}</span>
      {/if}
    </div>

    <MarkedSparql segments={attempt.segments} />

    {#if badge}
      <button
        type="button"
        class="constraint-badge"
        class:constraint-badge--ok={badge.ok}
        on:click={() => (constraintOpen = !constraintOpen)}
        title="Show the constraint query"
      >
        {badge.text} {constraintOpen ? '▲' : '▼'}
      </button>
    {/if}

    {#if constraintOpen && attempt.constraint}
      <div class="constraint">
        {#if attempt.constraint.sparql}
          <div class="constraint-label">
            Constraint query: candidates for {circled(attempt.step)}, given the placeholders resolved so far
          </div>
          <pre class="constraint-code">{prettyConstraint(attempt.constraint.sparql)}</pre>
        {/if}
        {#if attempt.constraint.reason}
          <div class="constraint-reason">Searched without constraint: {attempt.constraint.reason}</div>
        {/if}
      </div>
    {/if}

    {#if attempt.alternatives.length > 0}
      <table class="alts">
        <thead>
          <tr>
            <th class="col-rank">#</th>
            <th>Alternative</th>
            {#if attempt.reranked}<th class="col-score">Score</th>{/if}
          </tr>
        </thead>
        <tbody>
          {#each attempt.alternatives as alt, j (j)}
            <tr
              class:row--selected={alt.selected}
              class:row--rejected={alt.rejected}
              class:row--none={alt.isNone}
              class:row--below={alt.belowNone}
            >
              <td class="col-rank">{alt.rank + 1}</td>
              <td
                class="alt-cell"
                class:alt-cell--expanded={expanded[j]}
                title={expanded[j] ? '' : 'Click to show everything'}
                on:click={() => (expanded = { ...expanded, [j]: !expanded[j] })}
              >
                {#if alt.selected}<span class="check">✓</span>{/if}
                {#if alt.isNone}
                  <em>{alt.label}</em>
                {:else}
                  <strong class="label">{alt.label || alt.identifier}</strong>
                  {#if alt.identifier && alt.label !== alt.identifier}
                    {#if href(alt)}
                      <a class="id" href={href(alt)} target="_blank" rel="noopener noreferrer" on:click|stopPropagation><code>{alt.identifier}</code></a>
                    {:else}
                      <code class="id">{alt.identifier}</code>
                    {/if}
                  {/if}
                  {#if alt.variant}<span class="variant">as {alt.variant}</span>{/if}
                  {#if alt.was !== null && alt.was !== undefined && alt.was !== alt.rank}<span class="was">search #{alt.was + 1}</span>{/if}
                  {#if aliasText(alt)}<span class="aliases">· {aliasText(alt)}</span>{/if}
                  {#each infos(alt) as info}<span class="info">{info}</span>{/each}
                {/if}
              </td>
              {#if attempt.reranked}<td class="col-score">{alt.score?.toFixed(3) ?? ''}</td>{/if}
            </tr>
          {/each}
        </tbody>
      </table>
    {:else}
      <p class="empty">No alternatives found.</p>
    {/if}

    {#each attempt.outcome as o, k (k)}
      {@const out = outcomeText(o)}
      <div class="outcome outcome--{out.kind}"><span class="outcome-icon">{out.icon}</span>{out.text}</div>
    {/each}
  </div>
</div>

<style>
  .attempt {
    display: grid;
    grid-template-columns: 2rem 1fr;
    gap: var(--spacing-sm);
  }

  .attempt-rail {
    display: flex;
    justify-content: center;
  }

  .step {
    font-size: 1.35rem;
    line-height: 1;
    color: var(--text-subtle);
  }

  .step--done {
    color: var(--color-uni-green);
  }

  .attempt-body {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-xs);
    min-width: 0;
    padding: var(--spacing-sm);
    border: 1px solid var(--border-default);
    border-radius: var(--radius-sm);
    background: rgba(52, 74, 154, 0.02);
  }

  .wording-of {
    font-size: 0.75rem;
    color: var(--text-subtle);
  }

  .type {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--text-subtle);
  }

  .id {
    margin-left: 0.3em;
    font-family: 'Fira Code', monospace;
    font-size: 0.75rem;
    color: var(--text-subtle);
  }

  a.id {
    color: var(--color-uni-blue);
    text-decoration: underline;
    text-decoration-color: rgba(52, 74, 154, 0.4);
  }

  a.id code {
    font-family: inherit;
  }

  .attempt-header {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: var(--spacing-sm);
    font-size: 0.82rem;
  }

  .attempt-title {
    color: var(--text-primary);
  }

  .revisit {
    padding: 0.05rem 0.45rem;
    border-radius: var(--radius-sm);
    background: rgba(190, 170, 60, 0.15);
    color: #7a6a12;
    font-size: 0.75rem;
  }

  .constraint-badge {
    align-self: flex-start;
    padding: 0.05rem 0.45rem;
    border-radius: var(--radius-sm);
    border: 1px solid rgba(100, 116, 139, 0.3);
    background: rgba(100, 116, 139, 0.08);
    color: #475569;
    font-size: 0.75rem;
    cursor: pointer;
  }

  .constraint-badge--ok {
    border-color: rgba(52, 74, 154, 0.3);
    background: rgba(52, 74, 154, 0.08);
    color: var(--color-uni-blue);
  }

  .timing {
    margin-left: auto;
    font-size: 0.72rem;
    color: var(--text-subtle);
    white-space: nowrap;
  }

  .constraint {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-xs);
    padding: var(--spacing-sm);
    border-left: 3px solid var(--color-uni-blue);
    background: rgba(52, 74, 154, 0.05);
    font-size: 0.8rem;
  }

  .constraint-label {
    color: var(--text-subtle);
  }

  .constraint-code {
    margin: 0;
    font-family: 'Fira Code', monospace;
    font-size: 0.76rem;
    white-space: pre-wrap;
    word-break: break-word;
  }

  .constraint-reason {
    color: #92400e;
  }

  .alts {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
  }

  .alts th {
    text-align: left;
    font-weight: 600;
    font-size: 0.72rem;
    color: var(--text-subtle);
    padding: 0.2rem 0.4rem;
    border-bottom: 1px solid var(--border-default);
  }

  .alts td {
    padding: 0.2rem 0.4rem;
    vertical-align: top;
    border-bottom: 1px solid rgba(0, 0, 0, 0.04);
  }

  .col-rank {
    width: 2rem;
    color: var(--text-subtle);
  }

  .col-score {
    width: 4.5rem;
    text-align: right !important;
    font-family: 'Fira Code', monospace;
    font-size: 0.75rem;
    color: var(--text-subtle);
  }

  .row--selected {
    background: rgba(0, 160, 130, 0.1);
  }

  .row--rejected {
    text-decoration: line-through;
    opacity: 0.6;
  }

  .row--none {
    background: rgba(52, 74, 154, 0.06);
  }

  .row--below {
    opacity: 0.7;
  }

  .check {
    color: var(--color-uni-green);
    margin-right: 0.3em;
  }

  .variant,
  .was {
    margin-left: 0.4em;
    font-size: 0.72rem;
    color: var(--text-subtle);
    font-weight: 400;
  }

  .variant {
    font-family: 'Fira Code', monospace;
  }

  .was {
    font-style: italic;
  }

  .alt-cell {
    max-width: 0;
    white-space: nowrap;
    overflow: hidden;
    /* fade out what does not fit */
    -webkit-mask-image: linear-gradient(90deg, #000 85%, transparent);
    mask-image: linear-gradient(90deg, #000 85%, transparent);
    cursor: pointer;
  }

  .alt-cell--expanded {
    white-space: normal;
    -webkit-mask-image: none;
    mask-image: none;
  }

  .aliases {
    margin-left: 0.4em;
    font-size: 0.78rem;
    color: var(--text-subtle);
  }

  /* infos as small boxes like in search results; one line unless the
     row is expanded */
  .info {
    display: inline-block;
    margin-left: 0.35em;
    padding: 0 0.45rem;
    border: 1px solid rgba(52, 74, 154, 0.18);
    border-radius: var(--radius-sm);
    background: rgba(52, 74, 154, 0.04);
    font-size: 0.75rem;
    line-height: 1.5;
    color: var(--text-primary);
  }

  .alt-cell--expanded .info {
    margin-top: 0.2rem;
    white-space: normal;
  }

  .empty {
    margin: 0;
    font-size: 0.82rem;
    color: var(--text-subtle);
    font-style: italic;
  }

  .outcome {
    display: flex;
    gap: 0.4em;
    font-size: 0.82rem;
    font-weight: 500;
  }

  .outcome-icon {
    width: 1em;
    text-align: center;
  }

  .outcome--ok {
    color: #047857;
  }

  .outcome--warn {
    color: #92400e;
  }

  .outcome--bad {
    color: #b91c1c;
  }
</style>
