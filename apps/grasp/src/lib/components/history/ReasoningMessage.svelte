<script>
  import MessageCard from './MessageCard.svelte';
  import MarkdownContent from '../common/MarkdownContent.svelte';

  export let message;

  const reasoning = message?.reasoning ?? '';
  const content = message?.content ?? '';
  const hasReasoning = Boolean(reasoning.trim());
  const hasContent = Boolean(content.trim());
</script>

<!-- a model turn can carry an empty message; don't render an empty card -->
{#if hasReasoning || hasContent}
  <MessageCard title="Reasoning" accent="var(--color-uni-blue)">
    {#if hasReasoning}
      <MarkdownContent className="reasoning-block" content={reasoning} />
    {/if}

    {#if hasReasoning && hasContent}
      <hr class="divider" />
    {/if}

    {#if hasContent}
      <MarkdownContent className="content-block" content={content} />
    {/if}
  </MessageCard>
{/if}

<style>
  .divider {
    border: none;
    border-top: 1px solid rgba(0, 0, 0, 0.08);
    margin: var(--spacing-xs) 0;
  }

  .markdown.reasoning-block,
  .markdown.content-block {
    gap: var(--spacing-xs);
  }
</style>
