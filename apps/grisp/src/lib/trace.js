// Turns the GRISP event stream into one trace per skeleton: a timeline of
// attempts and backtracks in the order GRISP made them, plus a card per
// placeholder in fill order summarizing what it was resolved to.

const MARKER = /<rep>([\s\S]*?)<\/rep>|<iri>([\s\S]*?)<\/iri>/g;

const FAIL_REASONS = {
  timeout: 'timed out',
  no_alternatives: 'no fitting alternatives',
  validation_failed: 'final query has no results',
};

export function circled(step) {
  return step < 20 ? String.fromCharCode(0x2460 + step) : `(${step + 1})`;
}

export function failReason(reason) {
  return FAIL_REASONS[reason] ?? reason ?? 'failed';
}

// text and placeholder segments of a skeleton or step query, in document
// order; placeholders get their skeleton position and fill step, given the
// fill order and how many placeholders are already filled (and thus no
// longer marked in the query)
export function parseMarked(sparql, order, filled = 0) {
  const done = new Set(order.slice(0, filled));
  const open = order.map((_, i) => i).filter((i) => !done.has(i));
  const segments = [];
  let last = 0;
  let k = 0;
  for (const m of sparql.matchAll(MARKER)) {
    if (m.index > last) segments.push({ kind: 'text', text: sparql.slice(last, m.index) });
    const current = m[1] !== undefined;
    const position = open[k++] ?? null;
    segments.push({
      kind: current ? 'current' : 'open',
      wordings: (current ? m[1] : m[2]).split('<sep>'),
      position,
      step: position === null ? null : order.indexOf(position),
    });
    last = m.index + m[0].length;
  }
  if (last < sparql.length) segments.push({ kind: 'text', text: sparql.slice(last) });
  return segments;
}

// alternatives in the order they were tried; the ranking is a list of
// [index, score] pairs with a null index for "none"
function rankAlternatives(sel) {
  const alternatives = sel.alternatives ?? [];
  if (!sel.ranking?.length) {
    return alternatives.map((alt, i) => ({ ...alt, rank: i, was: null, score: null, isNone: false, belowNone: false }));
  }
  const items = [];
  let belowNone = false;
  sel.ranking.forEach(([index, score], rank) => {
    if (index === null) {
      items.push({ label: 'none of these', rank, score, isNone: true, belowNone: false });
      belowNone = true;
    } else if (index < alternatives.length) {
      items.push({ ...alternatives[index], rank, was: index, score, isNone: false, belowNone });
    }
  });
  return items;
}

function newSkeleton(entry, idx) {
  const segments = parseMarked(entry.skeleton, entry.order);
  const placeholders = segments.filter((s) => s.kind !== 'text');
  return {
    idx,
    segments,
    order: entry.order,
    status: 'pending',
    reason: null,
    cards: entry.order.map((position, step) => ({
      step,
      position,
      wordings: placeholders[position]?.wordings ?? [],
      attempts: [],
      selected: null,
    })),
    timeline: [],
    lastStep: 0,
  };
}

export function buildTrace(steps, finished) {
  const gen = steps.find((s) => s.type === 'skeletons');
  if (!gen) return null;

  const skeletons = gen.skeletons.map(newSkeleton);
  const total = skeletons.length;
  // what the server is working on next, for the live badge
  let cursor = { skeleton: 0, step: 0, phase: 'search' };

  for (const ev of steps) {
    if (ev.type !== 'selection') continue;
    const sk = skeletons[ev.skeleton];
    if (!sk) continue;
    const sel = ev.selection;
    if (sk.status === 'pending') sk.status = 'running';
    const card = sk.cards[sk.lastStep];
    const attempt = card?.attempts.at(-1);

    if (sel.type === 'alternatives') {
      const next = sk.cards[sel.index];
      const prev = next.attempts.at(-1);
      // nothing happened after the previous attempt: its wording was used up
      const nextWording = prev && prev.outcome.length === 0 && !prev.closed;
      if (nextWording) prev.outcome.push({ type: 'next_wording' });
      // selections of the earlier placeholders, to show which changed since the last attempt
      const context = sk.cards.slice(0, sel.index).map((c) => c.selected);
      const changed = prev
        ? context.flatMap((s, i) => (s?.identifier !== prev.context[i]?.identifier
          ? [`${circled(i)} = ${s?.label ?? s?.identifier}`]
          : []))
        : [];
      const attempt = {
        step: sel.index,
        wordings: next.wordings,
        wording: sel.query_index ?? 0,
        query: sel.query,
        segments: parseMarked(sel.sparql ?? '', sk.order, sel.index),
        alternatives: rankAlternatives(sel),
        reranked: !!sel.ranking?.length,
        constraint: sel.constraint ?? null,
        timings: sel.timings ?? null,
        context,
        changed: changed.join(', '),
        outcome: [],
        closed: false,
      };
      next.attempts.push(attempt);
      sk.timeline.push({ kind: 'attempt', attempt });
      sk.lastStep = sel.index;
      cursor = { skeleton: ev.skeleton, step: sel.index, phase: 'select' };
      continue;
    }

    if (attempt) attempt.closed = true;

    if (sel.type === 'select') {
      attempt?.outcome.push(sel);
      card.selected = sel;
      const item = attempt?.alternatives.find((a) => a.identifier === sel.identifier);
      if (item) item.selected = true;
      const step = sk.lastStep + 1;
      cursor = { skeleton: ev.skeleton, step, phase: step < sk.cards.length ? 'search' : 'validate' };
    } else if (sel.type === 'continue') {
      attempt?.outcome.push(sel);
      const item = attempt?.alternatives.find((a) => a.identifier === sel.identifier);
      if (item) item.rejected = true;
      cursor = { skeleton: ev.skeleton, step: sk.lastStep, phase: 'select' };
    } else if (sel.type === 'backtrack') {
      // the last selection is undone and its placeholder tried again
      const validation = sel.reason === 'validation_failed';
      const step = validation ? sk.cards.length - 1 : sk.lastStep - 1;
      if (sk.cards[step]) sk.cards[step].selected = null;
      sk.timeline.push({
        kind: 'event',
        tone: 'warn',
        text: validation
          ? `↩ Back to ${circled(step)} to try its next alternative`
          : `↩ No fitting alternative left for ${circled(sk.lastStep)}, back to ${circled(step)} to try its next alternative`,
      });
      cursor = { skeleton: ev.skeleton, step, phase: 'backtrack' };
    } else if (sel.type === 'validation') {
      const passed = sel.result === 'passed';
      sk.timeline.push({
        kind: 'event',
        tone: passed ? 'ok' : 'bad',
        text: passed ? '✓ The final query has results' : '✗ The final query has no results',
      });
      if (passed) {
        sk.status = 'resolved';
        cursor = { skeleton: ev.skeleton, step: sk.lastStep, phase: 'finish' };
      }
    } else if (sel.type === 'fail') {
      sk.status = 'failed';
      sk.reason = failReason(sel.reason);
      sk.timeline.push({ kind: 'event', tone: 'bad', text: `✗ Skeleton abandoned: ${sk.reason}` });
      cursor = ev.skeleton + 1 < total
        ? { skeleton: ev.skeleton + 1, step: 0, phase: 'search' }
        : { skeleton: ev.skeleton, step: 0, phase: 'finish' };
    }
  }

  // without the emptiness check there is no validation event
  for (const sk of skeletons) {
    if (sk.status === 'running' && finished && sk.cards.every((c) => c.selected)) sk.status = 'resolved';
  }

  const answer = skeletons.find((sk) => sk.status === 'resolved')?.idx ?? null;
  return { fillOrder: gen.fill_order ?? null, skeletons, answer, cursor };
}

export function liveStatus(trace) {
  if (!trace) return 'Generating skeletons...';
  const { skeleton, step, phase } = trace.cursor;
  const sk = trace.skeletons[skeleton];
  if (!sk || phase === 'finish') return 'Executing the final query...';
  const prefix = `Skeleton ${skeleton + 1} of ${trace.skeletons.length}`;
  if (phase === 'validate') return `${prefix} · checking the final query...`;
  const card = sk.cards[step];
  const name = card ? `${circled(step)} ${card.wordings[0] ?? ''}` : '';
  if (phase === 'backtrack') return `${prefix} · ↩ back to ${name}`;
  return `${prefix} · ${name}`;
}

const FILL_ORDERS = {
  'left-to-right': 'left to right',
  'right-to-left': 'right to left',
  random: 'random',
  'entities-then-properties': 'entities first, then properties',
  'properties-then-entities': 'properties first, then entities',
  'triple-wise-entities-then-properties': 'triple by triple, entities before properties',
};

export function formatFillOrder(fillOrder) {
  return FILL_ORDERS[fillOrder] ?? fillOrder;
}
