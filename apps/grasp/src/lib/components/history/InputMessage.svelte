<script>
  import MessageCard from './MessageCard.svelte';
  import MarkdownContent from '../common/MarkdownContent.svelte';
  export let message;

  $: elInput =
    message?.elInput &&
    typeof message.elInput === 'object' &&
    typeof message.elInput.data === 'string'
      ? message.elInput
      : null;
  $: elData = elInput?.data ?? '';
  $: elFrom = Number.isInteger(elInput?.annotate_from)
    ? Math.max(0, Math.min(elInput.annotate_from, elData.length))
    : null;
  $: elUpTo = Number.isInteger(elInput?.annotate_up_to)
    ? Math.max(0, Math.min(elInput.annotate_up_to, elData.length))
    : null;
  $: elHasWindow =
    elInput !== null &&
    elFrom !== null &&
    elUpTo !== null &&
    elUpTo > elFrom &&
    (elFrom > 0 || elUpTo < elData.length);
  $: elInstructions =
    typeof elInput?.special_instructions === 'string'
      ? elInput.special_instructions.trim()
      : '';

  $: rawInput = message?.input;
  $: inputText =
    typeof rawInput === 'string'
      ? rawInput
      : typeof rawInput?.input === 'string'
        ? rawInput.input
        : '';
  $: images = Array.isArray(rawInput?.image_input)
    ? rawInput.image_input.filter((entry) => typeof entry === 'string' && entry)
    : Array.isArray(rawInput?.image_url)
      ? rawInput.image_url.filter((entry) => typeof entry === 'string' && entry)
      : [];
  $: audio = Array.isArray(rawInput?.audio_input)
    ? rawInput.audio_input.filter((entry) => typeof entry === 'string' && entry)
    : [];
</script>

<MessageCard title="Input" accent="var(--color-uni-green)">
  {#if elInput}
    <div class="el-input-text">{#if elHasWindow}<span class="el-input-context">{elData.slice(0, elFrom)}</span><mark class="el-input-window">{elData.slice(elFrom, elUpTo)}</mark><span class="el-input-context">{elData.slice(elUpTo)}</span>{:else}{elData}{/if}</div>
    {#if elHasWindow}
      <p class="el-input-note">
        Only the highlighted part is annotated, the rest is used as context.
      </p>
    {/if}
    {#if elInstructions}
      <p class="el-input-instructions">
        <strong>Special instructions:</strong>
        {elInstructions}
      </p>
    {/if}
  {:else}
    <MarkdownContent content={inputText} />
    {#if images.length > 0 || audio.length > 0}
      <details class="input-evidence">
        <summary>Attached multimodal evidence</summary>
        {#if images.length > 0}
          <div class="input-images" aria-label="Submitted images">
            {#each images as image, index (`${index}-${image.slice(0, 64)}`)}
              <img src={image} alt={`Submitted image ${index + 1}`} loading="lazy" />
            {/each}
          </div>
        {/if}
        {#if audio.length > 0}
          <div class="input-audio" aria-label="Submitted audio">
            {#each audio as source, index (`${index}-${source.slice(0, 64)}`)}
              <audio src={source} controls preload="metadata">
                Your browser does not support audio playback.
              </audio>
            {/each}
          </div>
        {/if}
      </details>
    {/if}
  {/if}
</MessageCard>

<style>
  .el-input-text {
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    line-height: 1.6;
    font-size: 0.95rem;
    color: var(--text-primary);
  }

  .el-input-context {
    color: var(--text-subtle);
    opacity: 0.65;
  }

  .el-input-window {
    background: rgba(52, 74, 154, 0.18);
    color: var(--text-primary);
    border-radius: 2px;
  }

  .el-input-note {
    margin: var(--spacing-xs) 0 0;
    font-size: 0.8rem;
    color: var(--text-subtle);
  }

  .el-input-instructions {
    margin: var(--spacing-xs) 0 0;
    font-size: 0.85rem;
    color: var(--text-primary);
  }

  .input-evidence {
    border: 1px solid rgba(0, 0, 0, 0.06);
    border-radius: var(--radius-sm);
    padding: var(--spacing-xs) var(--spacing-sm);
    background: rgba(48, 127, 69, 0.035);
    margin-top: var(--spacing-sm);
  }

  .input-evidence summary {
    cursor: pointer;
    font-weight: 600;
    color: var(--color-uni-green);
    font-size: 0.85rem;
    list-style: none;
  }

  .input-evidence summary::-webkit-details-marker,
  .input-evidence summary::marker {
    display: none;
  }

  .input-evidence summary::before {
    content: '▸';
    display: inline-block;
    margin-right: var(--spacing-xs);
    transform: rotate(0deg);
    transition: transform 0.2s ease;
  }

  .input-evidence[open] summary::before {
    transform: rotate(90deg);
  }

  .input-images {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(8rem, 12rem));
    gap: var(--spacing-sm);
    margin-top: var(--spacing-sm);
  }

  .input-images img {
    width: 100%;
    max-height: 12rem;
    object-fit: contain;
    border-radius: 0.5rem;
  }

  .input-audio {
    display: grid;
    gap: var(--spacing-xs);
    margin-top: var(--spacing-sm);
  }

  .input-audio audio {
    width: 100%;
  }
</style>
