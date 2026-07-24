<script>
  import MessageCard from './MessageCard.svelte';
  import MarkdownContent from '../common/MarkdownContent.svelte';
  import SparqlBlock from '../common/SparqlBlock.svelte';
  import AnalyzeResultView from '../common/AnalyzeResultView.svelte';
  import { flattenFunctionArgs, toPreviewText, truncatePreview } from '../../utils/formatters.js';

  export let message;

  const COLLAPSED_TOOL_NAMES = new Set(['analyze', 'load']);
  const PRIMARY_INPUT_KEYS = ['input', 'user_input', 'url', 'uri', 'query', 'text', 'prompt'];
  const ARG_PREVIEW_MAX = 40;

  let showExtraArgs = false;
  let sparql = null;
  let analyzeEntries = [];
  let loadImagePayload = null;
  let showLoadImageLinkFallback = false;
  let activeLoadImageUrl = '';

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
        fullValue: toPreviewText(entry.value),
        previewValue: truncatePreview(toPreviewText(entry.value), ARG_PREVIEW_MAX)
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

  $: analyzeEntries = parseAnalyzeResults(toolName, message?.result);
  $: loadImagePayload = parseLoadImageResult(toolName, message?.result);
  $: {
    const nextUrl = loadImagePayload?.url ?? '';
    if (nextUrl !== activeLoadImageUrl) {
      activeLoadImageUrl = nextUrl;
      showLoadImageLinkFallback = false;
    }
  }

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

  function parseAnalyzeResults(name, value) {
    if (name !== 'analyze' || typeof value !== 'string') return [];
    const trimmed = value.trim();
    if (!trimmed) return [];

    try {
      const parsed = JSON.parse(trimmed);
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return [];

      return Object.entries(parsed).map(([modelName, modelValue]) => {
        if (modelValue && typeof modelValue === 'object' && !Array.isArray(modelValue)) {
          return {
            modelName,
            kind: 'structured',
            payload: modelValue
          };
        }

        return {
          modelName,
          kind: 'markdown',
          markdown: typeof modelValue === 'string' ? modelValue : String(modelValue ?? '')
        };
      });
    } catch {
      return [];
    }
  }

  function parseLoadImageResult(name, value) {
    if (name !== 'load' || typeof value !== 'string') return null;
    const trimmed = value.trim();
    if (!trimmed) return null;

    try {
      const parsed = JSON.parse(trimmed);
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) return null;
      if (parsed.type !== 'image_url') return null;

      const url = parsed?.image_url?.url;
      if (typeof url !== 'string' || !url.trim()) return null;

      return { url: url.trim() };
    } catch {
      return null;
    }
  }

  function handleLoadImageError() {
    showLoadImageLinkFallback = true;
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
          <span class="arg-chip" title={`${primaryArgChip.key}: ${primaryArgChip.fullValue}`}>
            <span class="arg-chip__key">{primaryArgChip.key}</span>
            <span class="arg-chip__value">{primaryArgChip.previewValue}</span>
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
          <span class="arg-chip" title={`${chip.key}: ${chip.fullValue}`}>
            <span class="arg-chip__key">{chip.key}</span>
            <span class="arg-chip__value">{chip.previewValue}</span>
          </span>
        {/each}
      {/if}
    </div>
  </svelte:fragment>

  {#if shouldCollapseArgs && showExtraArgs && hiddenArgChips.length > 0}
    <div class="arg-details" aria-label="Additional function arguments">
      {#each hiddenArgChips as chip (chip.id)}
        <span class="arg-chip arg-chip--linebreak" title={`${chip.key}: ${chip.fullValue}`}>
          <span class="arg-chip__key">{chip.key}</span>
          <span class="arg-chip__value">{chip.previewValue}</span>
        </span>
      {/each}
    </div>
  {/if}

  {#if sparql}
    <SparqlBlock code={sparql} qleverLink={qleverLink} label="SPARQL" />
  {/if}

  {#if analyzeEntries.length > 0}
    {#each analyzeEntries as entry, index (`${entry.modelName}-${index}`)}
      {#if entry.kind === 'structured'}
        <AnalyzeResultView payload={entry.payload} modelName={entry.modelName} raw={index === 0 ? message.result : ''} />
      {:else}
        <section class="analyze-markdown">
          <header class="analyze-markdown__header">
            {#if entry.modelName}
              <span class="analyze-markdown__model-chip">{entry.modelName}</span>
            {/if}
          </header>
          <MarkdownContent content={entry.markdown} />
        </section>
      {/if}
    {/each}
  {:else if loadImagePayload}
    <section class="load-image-preview" aria-label="Loaded image preview">
      {#if !showLoadImageLinkFallback}
        <img
          class="load-image-preview__image"
          src={loadImagePayload.url}
          alt="Loaded entity preview"
          loading="lazy"
          on:error={handleLoadImageError}
        />
      {:else}
        <a
          class="load-image-preview__link"
          href={loadImagePayload.url}
          target="_blank"
          rel="noopener noreferrer"
        >
          Open image source
        </a>
      {/if}
    </section>
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

  .analyze-markdown {
    display: grid;
    gap: var(--spacing-sm);
    padding: var(--spacing-md);
    border: 1px solid rgba(52, 74, 154, 0.15);
    border-radius: var(--radius-md);
    background: linear-gradient(180deg, rgba(52, 74, 154, 0.04), rgba(255, 255, 255, 0.9));
  }

  .analyze-markdown__header {
    display: flex;
    align-items: center;
    justify-content: flex-end;
  }

  .analyze-markdown__model-chip {
    border: 1px solid rgba(52, 74, 154, 0.22);
    border-radius: 999px;
    padding: 0.15rem 0.55rem;
    font-size: 0.72rem;
    color: var(--color-uni-blue);
    background: rgba(52, 74, 154, 0.08);
    white-space: nowrap;
  }

  .load-image-preview {
    display: grid;
    gap: var(--spacing-sm);
    padding: var(--spacing-sm);
    border: 1px solid rgba(190, 170, 60, 0.28);
    border-radius: var(--radius-md);
    background: linear-gradient(180deg, rgba(190, 170, 60, 0.06), rgba(255, 255, 255, 0.92));
  }

  .load-image-preview__image {
    display: block;
    width: min(100%, 640px);
    max-width: 100%;
    height: auto;
    border-radius: var(--radius-sm);
    border: 1px solid rgba(0, 0, 0, 0.08);
    background: #fff;
  }

  .load-image-preview__link {
    color: var(--color-uni-blue);
    text-decoration: underline;
    text-decoration-color: rgba(52, 74, 154, 0.45);
    width: fit-content;
    max-width: 100%;
    overflow-wrap: anywhere;
    word-break: break-word;
  }

  @media (max-width: 720px) {
    .arg-chip--linebreak {
      max-width: min(100%, 420px);
    }
  }
</style>
