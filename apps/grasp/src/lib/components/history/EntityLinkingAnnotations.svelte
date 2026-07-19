<script>
  export let text = '';
  export let annotateFrom = null;
  export let annotateUpTo = null;
  export let predictions = [];

  const NIL = '<NIL>';

  function normalize(value) {
    if (typeof value !== 'string') return '';
    let normalized = value;
    try {
      normalized = normalized.normalize('NFC');
    } catch (error) {
      console.warn('Failed to NFC-normalize text', error);
    }
    return normalized.replace(/[‘’]/g, "'");
  }

  function isNil(span) {
    return span?.entity === NIL || span?.identifier === NIL;
  }

  function entityHref(span) {
    if (typeof span?.identifier === 'string') {
      const cleaned = span.identifier.replace(/^<|>$/g, '').trim();
      if (/^https?:\/\//.test(cleaned)) return cleaned;
    }
    if (
      typeof span?.entity === 'string' &&
      span.entity.startsWith('wd:')
    ) {
      return `https://www.wikidata.org/wiki/${span.entity.slice(3)}`;
    }
    return null;
  }

  function spanDescription(span) {
    if (isNil(span)) return 'Not linked to any entity (NIL)';
    const label = typeof span.label === 'string' ? span.label.trim() : '';
    return label ? `${label} — ${span.entity}` : span.entity;
  }

  function buildSpans(value, preds) {
    const spans = [];
    for (const [index, pred] of (Array.isArray(preds) ? preds : []).entries()) {
      const start = pred?.start_char;
      const end = pred?.end_char;
      if (!Number.isInteger(start) || !Number.isInteger(end)) continue;
      if (start < 0 || end <= start || end > value.length) continue;
      spans.push({
        id: index,
        start,
        end,
        entity:
          typeof pred.entity_reference === 'string'
            ? pred.entity_reference
            : '',
        identifier:
          typeof pred.identifier === 'string' ? pred.identifier : null,
        label: typeof pred.label === 'string' ? pred.label : null
      });
    }
    return spans;
  }

  function buildSegments(value, spanList, winStart, winEnd) {
    const cuts = new Set([0, value.length]);
    for (const span of spanList) {
      cuts.add(span.start);
      cuts.add(span.end);
    }
    if (winStart !== null) cuts.add(winStart);
    if (winEnd !== null) cuts.add(winEnd);
    const points = [...cuts]
      .filter((p) => p >= 0 && p <= value.length)
      .sort((a, b) => a - b);

    const segments = [];
    for (let i = 0; i < points.length - 1; i += 1) {
      const segStart = points[i];
      const segEnd = points[i + 1];
      if (segEnd <= segStart) continue;
      const covering = spanList.filter(
        (span) => span.start <= segStart && span.end >= segEnd
      );
      // innermost (latest-starting, shortest) span wins for the link
      covering.sort((a, b) => b.start - a.start || a.end - b.end);
      const primary = covering[0] ?? null;
      const context =
        (winStart !== null && segEnd <= winStart) ||
        (winEnd !== null && segStart >= winEnd);
      segments.push({
        text: value.slice(segStart, segEnd),
        primary,
        nested: covering.length > 1,
        nil: primary ? isNil(primary) : false,
        href: primary && !isNil(primary) ? entityHref(primary) : null,
        title: covering.map(spanDescription).join('\n'),
        context
      });
    }
    return segments;
  }

  $: normalizedText = normalize(text);
  $: spans = buildSpans(normalizedText, predictions);
  $: windowStart =
    Number.isInteger(annotateFrom) && annotateFrom > 0
      ? Math.min(annotateFrom, normalizedText.length)
      : null;
  $: windowEnd =
    Number.isInteger(annotateUpTo) && annotateUpTo < normalizedText.length
      ? Math.max(annotateUpTo, 0)
      : null;
  $: hasWindow = windowStart !== null || windowEnd !== null;
  $: segments = buildSegments(normalizedText, spans, windowStart, windowEnd);

  $: linkedSpans = spans.filter((span) => !isNil(span));
  $: nilCount = spans.length - linkedSpans.length;
  $: distinctEntities = new Set(
    linkedSpans.map((span) => span.identifier ?? span.entity)
  ).size;
  $: summaryParts = [
    `${spans.length} annotated span${spans.length === 1 ? '' : 's'}`,
    `${distinctEntities} distinct entit${distinctEntities === 1 ? 'y' : 'ies'}`,
    ...(nilCount > 0
      ? [`${nilCount} not linked (NIL)`]
      : [])
  ];
</script>

<div class="el-result">
  <div class="el-text">{#each segments as segment, index (index)}{#if segment.primary && segment.href}<a
        class="el-span"
        class:el-span--nested={segment.nested}
        class:el-span--context={segment.context}
        href={segment.href}
        target="_blank"
        rel="noopener noreferrer"
        title={segment.title}
      >{segment.text}</a>{:else if segment.primary}<span
        class="el-span"
        class:el-span--nil={segment.nil}
        class:el-span--nested={segment.nested}
        class:el-span--context={segment.context}
        title={segment.title}
      >{segment.text}</span>{:else if segment.context}<span
        class="el-context">{segment.text}</span>{:else}{segment.text}{/if}{/each}</div>

  <p class="el-summary">
    {summaryParts.join(' · ')}
    {#if hasWindow}
      · dimmed text is context outside the annotation window
    {/if}
  </p>
</div>

<style>
  .el-result {
    display: grid;
    gap: var(--spacing-sm);
  }

  .el-text {
    border: 1px solid rgba(52, 74, 154, 0.2);
    border-radius: var(--radius-sm);
    background: #fff;
    padding: var(--spacing-md);
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    line-height: 1.7;
    font-size: 0.95rem;
    color: var(--text-primary);
  }

  .el-span {
    background: rgba(52, 74, 154, 0.14);
    border-bottom: 2px solid rgba(52, 74, 154, 0.55);
    border-radius: 3px 3px 0 0;
    padding: 0.08rem 0.15rem;
    color: var(--color-uni-blue);
    font-weight: 600;
    text-decoration: none;
  }

  a.el-span:hover {
    background: rgba(52, 74, 154, 0.24);
    text-decoration: none;
  }

  .el-span--nested {
    box-shadow: 0 2px 0 0 rgba(163, 83, 148, 0.6);
  }

  .el-span--nil {
    background: rgba(180, 180, 180, 0.25);
    border-bottom: 2px dashed rgba(120, 120, 120, 0.7);
    color: var(--text-subtle);
    cursor: help;
  }

  .el-context,
  .el-span--context {
    color: var(--text-subtle);
    opacity: 0.65;
  }

  .el-summary {
    margin: 0;
    font-size: 0.85rem;
    color: var(--text-subtle);
  }
</style>
