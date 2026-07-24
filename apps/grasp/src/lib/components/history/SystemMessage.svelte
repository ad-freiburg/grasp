<script>
  import MessageCard from './MessageCard.svelte';
  import MarkdownContent from '../common/MarkdownContent.svelte';
  import { prettyJson } from '../../utils/formatters.js';

  export let message;

  let functionsMarkdown = '';
  $: modelName = (() => {
    const config = message?.config;
    const configuredModels = Array.isArray(config?.models) ? config.models : [];

    const graspModel = configuredModels.find((entry) => {
      const modalities = Array.isArray(entry?.modality) ? entry.modality : [];
      return modalities.some(
        (value) => typeof value === 'string' && value.toLowerCase() === 'grasp'
      );
    });

    if (typeof graspModel?.model === 'string') {
      const model = graspModel.model.trim();
      if (model) {
        return model;
      }
    }

    return typeof config?.model === 'string' ? config.model.trim() : '';
  })();

  $: functionsMarkdown = Array.isArray(message?.functions)
    ? message.functions
        .map(
          (fn) =>
            `**${fn.name}**\n\n${fn.description ?? ''}\n\n*JSON Schema*\n\n\u0060\u0060\u0060json\n${prettyJson(
              fn.parameters
            )}\n\u0060\u0060\u0060`
        )
        .join('\n\n')
    : '';
</script>

<MessageCard title="System" accent="var(--color-uni-pink)">
  <svelte:fragment slot="meta">
    {#if modelName}
      <span class="model-chip">Model · {modelName}</span>
    {/if}
  </svelte:fragment>

  {#if functionsMarkdown}
    <details>
      <summary>Functions</summary>
      <MarkdownContent content={functionsMarkdown} />
    </details>
  {/if}

  {#if message?.system_message}
    <details>
      <summary>GRASP instruction</summary>
      <MarkdownContent content={message.system_message} />
    </details>
  {/if}
</MessageCard>

<style>
  details {
    border: 1px solid rgba(0, 0, 0, 0.06);
    border-radius: var(--radius-sm);
    padding: var(--spacing-xs) var(--spacing-sm);
    background: rgba(163, 83, 148, 0.035);
    margin: 0;
  }

  summary {
    cursor: pointer;
    font-weight: 600;
    color: var(--color-uni-pink);
    font-size: 0.85rem;
    list-style: none;
  }

  summary::-webkit-details-marker,
  summary::marker {
    display: none;
  }

  summary::before {
    content: '▸';
    display: inline-block;
    margin-right: var(--spacing-xs);
    transform: rotate(0deg);
    transition: transform 0.2s ease;
  }

  details[open] summary::before {
    transform: rotate(90deg);
  }

  details + details {
    margin-top: var(--spacing-xs);
  }

  .model-chip {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-xs);
    padding: 0.2rem 0.75rem;
    border-radius: 999px;
    background: rgba(163, 83, 148, 0.12);
    color: var(--color-uni-pink);
    font-size: 0.75rem;
    font-weight: 600;
  }
</style>
