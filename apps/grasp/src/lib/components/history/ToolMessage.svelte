<script>
  import MessageCard from './MessageCard.svelte';
  import MarkdownContent from '../common/MarkdownContent.svelte';
  import SparqlBlock from '../common/SparqlBlock.svelte';
  import AnalyzeResultView from '../common/AnalyzeResultView.svelte';
  import { flattenFunctionArgs } from '../../utils/formatters.js';

  export let message;

  const COLLAPSED_TOOL_NAMES = new Set(['analyze', 'load']);
  const PRIMARY_INPUT_KEYS = ['input', 'user_input', 'url', 'uri', 'query', 'text', 'prompt'];

  let showExtraArgs = false;
  let sparql = null;
  let analyzeData = null;
  let analyzeModelName = '';

  $: toolName = typeof message?.name === 'string' ? message.name : '';
  $: shouldCollapseArgs = COLLAPSED_TOOL_NAMES.has(toolName);
  $: flattened = flattenFunctionArgs(message?.args ?? {});

  let argChips = [];
  $: {
    sparql = null;
    argChips = [];
    for (const [index, entry] of flattened.entries()) {
      if (entry.key === 'sparql') {
        sparql = entry.value;
        continue;
      }
      argChips.push({
        id: `${entry.key}-${index}`,
        key: entry.key,
        value: coerceArgValue(entry.value)
      });
    }
  }

  $: primaryArgChipId = shouldCollapseArgs ? pickPrimaryArgChipId(argChips) : null;
  $: primaryArgChip =
    shouldCollapseArgs && primaryArgChipId
      ? argChips.find((chip) => chip.id === primaryArgChipId) ?? null
      : null;
  $: hiddenArgChips =
    shouldCollapseArgs && primaryArgChip
      ? argChips.filter((chip) => chip.id !== primaryArgChip.id)
      : shouldCollapseArgs
        ? [...argChips]
        : [];
  $: if (!shouldCollapseArgs) {
    showExtraArgs = false;
  }

  $: analyzeParseResult = parseAnalyzeResult(toolName, message?.result);
  $: analyzeData = analyzeParseResult?.payload ?? null;
  $: analyzeModelName = analyzeParseResult?.modelName ?? '';

  const qleverLink = null;

  function pickPrimaryArgChipId(chips) {
    if (!Array.isArray(chips) || chips.length === 0) return null;
    for (const preferredKey of PRIMARY_INPUT_KEYS) {
      const match = chips.find(
        (chip) => chip.key === preferredKey || chip.key.endsWith(`.${preferredKey}`)
      );
      if (match) return match.id;
    }
    return chips[0].id;
  }

  function coerceArgValue(value) {
    if (value === null || value === undefined) {
      return '';
    }
    if (typeof value === 'string') {
      return normalizeWhitespace(value);
    }
    if (typeof value === 'number' || typeof value === 'boolean') {
      return String(value);
    }
    try {
      return normalizeWhitespace(JSON.stringify(value));
    } catch (error) {
      console.warn('Failed to stringify function argument', error);
      return normalizeWhitespace(String(value));
    }
  }

  function normalizeWhitespace(text) {
    return text.replace(/\s+/g, ' ').trim();
  }

  function parseAnalyzeResult(name, value) {
    if (name !== 'analyze' || typeof value !== 'string') return null;
    const trimmed = value.trim();
    if (!trimmed) return null;

    try {
      const parsed = JSON.parse(trimmed);
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
      const [firstModelName] = Object.keys(parsed);
      if (!firstModelName) return null;
      const payload = parsed[firstModelName];
      if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return null;
      return { payload, modelName: firstModelName };
    } catch {
      return null;
    }
  }
</script>

<MessageCard title="Function Call" accent="var(--color-uni-yellow)">
  <svelte:fragment slot="meta">
    <div class="meta-group">
      {#if message?.name}
        <span class="function-chip">
          <span class="function-chip__key">function</span>
          <span class="function-chip__value">{message.name}</span>
        </span>
      {/if}

      {#if shouldCollapseArgs}
        {#if primaryArgChip}
          <span class="arg-chip" title={`${primaryArgChip.key}: ${primaryArgChip.value}`}>
            <span class="arg-chip__key">{primaryArgChip.key}</span>
            <span class="arg-chip__value">{primaryArgChip.value}</span>
          </span>
        {/if}

        {#if hiddenArgChips.length > 0}
          <button
            type="button"
            class="arg-toggle"
            on:click={() => (showExtraArgs = !showExtraArgs)}
            aria-expanded={showExtraArgs}
            aria-label={showExtraArgs ? 'Show fewer attributes' : 'Show more attributes'}
            title={showExtraArgs ? 'Show fewer attributes' : 'Show more attributes'}
          >
            {#if showExtraArgs}
              ▲
            {:else}
              ▼
            {/if}
          </button>
        {/if}
      {:else}
        {#each argChips as chip (chip.id)}
          <span class="arg-chip" title={`${chip.key}: ${chip.value}`}>
            <span class="arg-chip__key">{chip.key}</span>
            <span class="arg-chip__value">{chip.value}</span>
          </span>
        {/each}
      {/if}
    </div>
  </svelte:fragment>

  {#if shouldCollapseArgs && showExtraArgs && hiddenArgChips.length > 0}
    <div class="arg-details" aria-label="Additional function arguments">
      {#each hiddenArgChips as chip (chip.id)}
        <span class="arg-chip arg-chip--linebreak" title={`${chip.key}: ${chip.value}`}>
          <span class="arg-chip__key">{chip.key}</span>
          <span class="arg-chip__value">{chip.value}</span>
        </span>
      {/each}
    </div>
  {/if}

  {#if sparql}
    <SparqlBlock code={sparql} qleverLink={qleverLink} label="SPARQL" />
  {/if}

  {#if analyzeData}
    <AnalyzeResultView payload={analyzeData} modelName={analyzeModelName} raw={message.result} />
  {:else if message?.result}
    <MarkdownContent content={message.result} />
  {/if}
</MessageCard>

<style>
  .function-chip {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-xs);
    padding: 0.25rem 0.7rem;
    border-radius: 999px;
    background: #fff;
    border: 1px solid rgba(190, 170, 60, 0.6);
    font-size: 0.75rem;
    white-space: nowrap;
  }

  .function-chip__key {
    font-weight: 700;
    color: var(--color-uni-yellow);
    text-transform: uppercase;
    letter-spacing: 0.03em;
  }

  .function-chip__value {
    color: var(--text-primary);
  }

  .arg-chip {
    display: inline-flex;
    align-items: flex-start;
    gap: var(--spacing-xs);
    padding: 0.25rem 0.65rem;
    border-radius: var(--radius-sm);
    background: #fff;
    border: 1px solid rgba(190, 170, 60, 0.6);
    font-size: 0.75rem;
    max-width: min(100%, 560px);
    min-width: 0;
  }

  .arg-chip__key {
    font-weight: 700;
    color: var(--color-uni-yellow);
    flex: 0 0 auto;
  }

  .arg-chip__value {
    color: var(--text-primary);
    min-width: 0;
    white-space: normal;
    overflow-wrap: anywhere;
    word-break: break-word;
  }

  .arg-toggle {
    appearance: none;
    border: 1px solid rgba(190, 170, 60, 0.5);
    background: rgba(190, 170, 60, 0.08);
    color: var(--text-primary);
    border-radius: var(--radius-sm);
    padding: 0.2rem 0.5rem;
    font-size: 0.72rem;
    font-weight: 600;
    cursor: pointer;
    text-transform: lowercase;
  }

  .arg-toggle:hover {
    background: rgba(190, 170, 60, 0.16);
  }

  .arg-details {
    margin-top: var(--spacing-xs);
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: var(--spacing-xs);
  }

  .arg-chip--linebreak {
    display: inline-flex;
    width: auto;
    max-width: 100%;
  }

  .arg-chip--linebreak .arg-chip__value {
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    overflow-wrap: normal;
    word-break: normal;
  }

  .arg-chip--linebreak .arg-chip__key {
    white-space: nowrap;
  }

  @media (max-width: 720px) {
    .arg-chip--linebreak {
      max-width: min(100%, 420px);
    }
  }
</style>
