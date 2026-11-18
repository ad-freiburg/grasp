<script>
  import MessageCard from './MessageCard.svelte';
  import MarkdownContent from '../common/MarkdownContent.svelte';

  export let message;
</script>

<MessageCard title="Limit reached" accent="var(--color-uni-red)">
  <MarkdownContent content={message?.content ?? ''} />
</MessageCard>
