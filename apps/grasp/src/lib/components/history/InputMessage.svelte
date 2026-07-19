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
    <MarkdownContent content={message?.input ?? ''} />
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
</style>
