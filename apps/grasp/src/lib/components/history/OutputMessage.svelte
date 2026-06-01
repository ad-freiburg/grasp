<script>
  import { tick } from 'svelte';
  import MessageCard from './MessageCard.svelte';
  import MarkdownContent from '../common/MarkdownContent.svelte';
  import SparqlBlock from '../common/SparqlBlock.svelte';
  import { prettyJson } from '../../utils/formatters.js';
  import { QLEVER_HOSTS, sharePathForId } from '../../constants.js';

  export let message;
  export let shareConversation = null;
  export let shareDisabled = false;

const output = message?.output ?? {};
const task = message?.task;
const elapsed = typeof message?.elapsed === 'number' ? message.elapsed : null;

let shareStatus = 'idle';
let shareLink = '';
let shareError = '';
let shareModalOpen = false;
let shareCopyError = '';
let shareLinkInputEl;

const shareModalTitleId = `share-modal-title-${Math.random().toString(36).slice(2, 8)}`;
const shareModalDescriptionId = `share-modal-description-${Math.random().toString(36).slice(2, 8)}`;

let primaryText = '';
if (task === 'sparql-qa') {
  primaryText =
    output?.type === 'answer'
      ? output?.answer ?? ''
      : output?.explanation ?? '';
} else if (task === 'sparql-to-question') {
  primaryText = output?.formatted ?? '';
} else if (task === 'general-qa') {
  primaryText = output?.output ?? '';
}

const sparql = task === 'sparql-to-question' ? null : output?.sparql ?? null;
const selections = output?.selections ?? null;
const result = output?.result ?? null;
const endpoint = output?.endpoint ?? null;
const ceaFormatted = task === 'cea' ? output?.formatted ?? '' : '';
const ceaInputTable =
  task === 'cea' && isValidCeaTable(message?.ceaInputTable)
    ? message.ceaInputTable
    : null;
const ceaAnnotations =
  task === 'cea' && Array.isArray(output?.annotations)
    ? output.annotations.filter(
        (item) => item && typeof item === 'object' && !Array.isArray(item)
      )
    : [];

const fence = '\u0060\u0060\u0060';

function toMarkdown(value, language = 'json') {
  if (value === null || value === undefined) return '';
  if (typeof value === 'string') return value;
  return `${fence}${language}\n${prettyJson(value)}\n${fence}`;
}

function cleanIdentifier(identifier) {
  if (typeof identifier !== 'string') return null;
  return identifier.replace(/^<|>$/g, '').trim() || null;
}

function isValidCeaTable(table) {
  if (!table || typeof table !== 'object') return false;
  if (!Array.isArray(table.header) || !Array.isArray(table.data)) return false;
  return true;
}

function annotationHref(annotation) {
  const cleaned = cleanIdentifier(annotation?.identifier);
  if (cleaned) return cleaned;
  if (typeof annotation?.entity === 'string' && annotation.entity.startsWith('wd:')) {
    return `https://www.wikidata.org/wiki/${annotation.entity.slice(3)}`;
  }
  return null;
}

function displayIndex(value) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value + 1;
  }
  if (value === null || value === undefined || value === '') return 'N/A';
  return value;
}

function displayCellValue(value) {
  if (value === null || value === undefined) return 'N/A';
  if (typeof value === 'string') return value;
  if (typeof value === 'number' && Number.isFinite(value)) return String(value);
  if (typeof value === 'boolean') return value ? 'true' : 'false';
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function getAnnotationValue(annotation) {
  if (!ceaInputTable) return null;
  const rowIndex = annotation?.row;
  const columnIndex = annotation?.column;
  if (!Number.isInteger(rowIndex) || rowIndex < 0) return null;
  if (!Number.isInteger(columnIndex) || columnIndex < 0) return null;

  const rows = Array.isArray(ceaInputTable.data) ? ceaInputTable.data : [];
  if (rowIndex >= rows.length) return null;

  const row = rows[rowIndex];
  if (!Array.isArray(row) || columnIndex >= row.length) return null;

  return row[columnIndex];
}

function deriveQleverLink() {
  if (!sparql || !endpoint) return null;
  try {
    const url = new URL(endpoint);
    if (!QLEVER_HOSTS.includes(url.host)) return null;
      const base = endpoint.replace('/api', '');
      const separator = base.includes('?') ? '&' : '?';
      return `${base}${separator}query=${encodeURIComponent(sparql)}&exec=true`;
    } catch (error) {
      console.warn('Failed to build QLever link', error);
      return null;
    }
  }

  const qleverLink = deriveQleverLink();

  async function handleShareClick() {
    if (shareDisabled || shareStatus === 'pending') return;
    if (typeof shareConversation !== 'function') return;
    shareError = '';

    if (shareLink) {
      await openShareModal();
      return;
    }

    shareStatus = 'pending';
    shareLink = '';
    try {
      const result = await shareConversation({ message });
      shareLink = sharePathForId(result?.id);
      if (!shareLink) {
        shareStatus = 'error';
        shareError = 'Share link unavailable.';
        return;
      }
      shareStatus = 'success';
      await openShareModal();
    } catch (error) {
      shareStatus = 'error';
      shareLink = '';
      shareError =
        error?.message && typeof error.message === 'string'
          ? error.message
          : 'Failed to create share link.';
    }
  }

  async function openShareModal() {
    if (!shareLink) return;
    shareModalOpen = true;
    resetShareCopyState();
    await tick();
    if (shareLinkInputEl && typeof shareLinkInputEl.select === 'function') {
      shareLinkInputEl.focus();
      shareLinkInputEl.select();
    }
  }

  function closeShareModal() {
    shareModalOpen = false;
    resetShareCopyState();
  }

  function resetShareCopyState() {
    shareCopyError = '';
  }

  function handleModalBackdropClick(event) {
    if (event.target !== event.currentTarget) return;
    closeShareModal();
  }

  function handleModalKeydown(event) {
    if (!shareModalOpen) return;
    if (event.key === 'Escape') {
      event.preventDefault();
      closeShareModal();
    }
  }

  async function handleShareCopy() {
    if (!shareLink) return;
    resetShareCopyState();

    try {
      if (
        typeof navigator !== 'undefined' &&
        navigator?.clipboard &&
        typeof navigator.clipboard.writeText === 'function'
      ) {
        await navigator.clipboard.writeText(shareLink);
      } else if (
        shareLinkInputEl &&
        typeof shareLinkInputEl.select === 'function'
      ) {
        shareLinkInputEl.focus();
        shareLinkInputEl.select();
        const successful =
          typeof document !== 'undefined' &&
          typeof document.execCommand === 'function' &&
          document.execCommand('copy');
        if (!successful) {
          throw new Error('execCommand failed');
        }
      } else {
        throw new Error('Clipboard unavailable');
      }
    } catch (error) {
      console.warn('Failed to copy share link', error);
      shareCopyError = 'Copy failed. Please copy the link manually.';
      if (shareLinkInputEl && typeof shareLinkInputEl.select === 'function') {
        shareLinkInputEl.focus();
        shareLinkInputEl.select();
      }
      return;
    }
  }
</script>

<svelte:window on:keydown={handleModalKeydown} />

<MessageCard title="Output" accent="var(--color-uni-dark-blue)">
  {#if task === 'cea'}
    {#if ceaFormatted}
      <section class="cea-section">
        <h3 class="cea-heading">Formatted Result</h3>
        <MarkdownContent content={ceaFormatted} />
      </section>
    {/if}

    {#if ceaAnnotations.length > 0}
      <section class="cea-section">
        <h3 class="cea-heading">Annotations</h3>
        <div class="cea-annotations">
          <table>
            <thead>
              <tr>
                <th scope="col">Row</th>
                <th scope="col">Column</th>
                <th scope="col">Value</th>
                <th scope="col">Annotation</th>
              </tr>
            </thead>
            <tbody>
              {#each ceaAnnotations as annotation, index (index)}
                <tr>
                  <td data-title="Row">{displayIndex(annotation?.row)}</td>
                  <td data-title="Column">{displayIndex(annotation?.column)}</td>
                  <td data-title="Value">{displayCellValue(getAnnotationValue(annotation))}</td>
                  <td data-title="Annotation">
                    {#if annotation?.label || annotation?.entity}
                      {annotation?.label ?? annotation?.entity ?? 'Unknown'}
                      {#if annotation?.entity}
                        {' ('}
                        {#if annotationHref(annotation)}
                          <a href={annotationHref(annotation)} target="_blank" rel="noopener noreferrer">
                            {annotation.entity}
                          </a>
                        {:else}
                          {annotation.entity}
                        {/if}
                        {')'}
                      {/if}
                    {:else}
                      Unknown
                    {/if}
                  </td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>
      </section>
    {/if}

    {#if !ceaFormatted && ceaAnnotations.length === 0}
      <p class="placeholder">No annotations returned.</p>
    {/if}
  {:else}
    {#if primaryText}
      <MarkdownContent content={primaryText} />
    {/if}

    {#if sparql}
      <SparqlBlock code={sparql} qleverLink={qleverLink} label="SPARQL" />
    {/if}

    {#if selections}
      <MarkdownContent content={toMarkdown(selections)} />
    {/if}

    {#if result}
      <MarkdownContent content={toMarkdown(result)} />
    {/if}

    {#if !primaryText && !sparql && !result}
      <p class="placeholder">No output generated.</p>
    {/if}
  {/if}

  <div class="footer">
    <div class="footer__left">
      <button
        type="button"
        class="share-button"
        class:share-button--pending={shareStatus === 'pending'}
        on:click={handleShareClick}
        disabled={shareStatus === 'pending' || shareDisabled}
        aria-disabled={shareDisabled}
        title={shareDisabled
          ? 'Sharing disabled for imported conversations'
          : 'Share conversation'}
      >
        {#if shareStatus === 'pending'}
          <span class="share-spinner" aria-hidden="true"></span>
          Generating link…
        {:else}
          <span class="share-icon" aria-hidden="true">⥂</span>
          {#if shareStatus === 'success'}
            Share link ready
          {:else}
            Share
          {/if}
        {/if}
      </button>
      {#if shareStatus === 'error'}
        <span class="share-error" role="alert">{shareError}</span>
      {/if}
    </div>
    {#if elapsed !== null}
      <div class="footer__right">
        <span class="chip">Took {elapsed.toFixed(2)} s</span>
      </div>
    {/if}
  </div>
</MessageCard>

{#if shareModalOpen}
  <div
    class="share-modal-backdrop"
    role="presentation"
    on:click={handleModalBackdropClick}
  >
    <div
      class="share-modal"
      role="dialog"
      aria-modal="true"
      aria-labelledby={shareModalTitleId}
      aria-describedby={shareModalDescriptionId}
      on:click|stopPropagation
    >
      <header class="share-modal__header">
        <h3 class="share-modal__title" id={shareModalTitleId}>
          Share Conversation
        </h3>
      </header>
      <div class="share-modal__body">
        <p class="share-modal__description" id={shareModalDescriptionId}>
          Share this link to reopen the conversation later.
        </p>
        <div class="share-modal__link">
          <input
            class="share-modal__input"
            type="text"
            value={shareLink}
            readonly
            spellcheck="false"
            aria-label="Share link"
            bind:this={shareLinkInputEl}
          />
          <button
            type="button"
            class="share-modal__copy-button"
            on:click={handleShareCopy}
            aria-label="Copy share link"
            title="Copy share link"
          >
            <span class="copy-icon" aria-hidden="true">⧉</span>
          </button>
        </div>
        {#if shareCopyError}
          <p class="share-modal__error" role="alert">{shareCopyError}</p>
        {/if}
      </div>
    </div>
  </div>
{/if}

<style>
  .chip {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-xs);
    padding: 0.2rem 0.75rem;
    border-radius: 999px;
    background: rgba(0, 1, 73, 0.15);
    color: var(--color-uni-dark-blue);
    font-size: 0.75rem;
    font-weight: 600;
  }

  .placeholder {
    margin: 0;
    color: var(--text-subtle);
    font-size: 0.9rem;
  }

  .footer {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: var(--spacing-sm);
    margin-top: var(--spacing-sm);
  }

  .footer__left {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-sm);
    flex: 1 1 auto;
    min-width: 240px;
  }

  .footer__right {
    display: inline-flex;
    align-items: center;
    justify-content: flex-end;
    flex: 0 0 auto;
  }

  .share-button {
    display: inline-flex;
    align-items: center;
    gap: var(--spacing-xs);
    padding: 0.35rem 0.85rem;
    border-radius: var(--radius-sm);
    border: 1px solid rgba(52, 74, 154, 0.28);
    background: rgba(52, 74, 154, 0.12);
    color: var(--color-uni-blue);
    font-weight: 600;
    cursor: pointer;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }

  .share-button:not(:disabled):hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 12px rgba(52, 74, 154, 0.16);
  }

  .share-button:disabled {
    opacity: 0.65;
    cursor: not-allowed;
    transform: none;
    box-shadow: none;
  }

  .share-button--pending {
    background: rgba(52, 74, 154, 0.18);
    color: var(--color-uni-blue);
  }

  .share-icon {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    line-height: 1;
  }

  .share-modal-backdrop {
    position: fixed;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1.5rem;
    background: rgba(0, 0, 0, 0.35);
    z-index: 1000;
  }

  .share-modal {
    width: min(100%, 420px);
    background: var(--surface, #fff);
    border-radius: var(--radius-md, 12px);
    box-shadow: 0 18px 40px rgba(0, 0, 0, 0.18);
    display: flex;
    flex-direction: column;
    gap: var(--spacing-md);
    padding: var(--spacing-lg);
  }

  .share-modal__header {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: var(--spacing-sm);
  }

  .share-modal__title {
    margin: 0;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--color-uni-dark-blue);
  }

  .share-modal__body {
    display: flex;
    flex-direction: column;
    gap: var(--spacing-sm);
  }

  .share-modal__description {
    margin: 0;
    color: var(--text-subtle);
    font-size: 0.9rem;
  }

  .share-modal__link {
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
  }

  .share-modal__input {
    flex: 1;
    padding: 0.45rem 0.6rem;
    border-radius: var(--radius-sm);
    border: 1px solid rgba(52, 74, 154, 0.35);
    font-family: var(--font-mono, 'Courier New', monospace);
    font-size: 0.9rem;
    color: var(--color-uni-dark-blue);
    background: rgba(52, 74, 154, 0.08);
  }

  .share-modal__copy-button {
    border: 1px solid rgba(52, 74, 154, 0.3);
    background: rgba(52, 74, 154, 0.12);
    border-radius: var(--radius-sm);
    padding: 0.35rem 0.55rem;
    cursor: pointer;
    color: var(--color-uni-blue);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
  }

  .copy-icon {
    font-size: 1rem;
    line-height: 1;
  }

  .share-modal__error {
    margin: 0;
    font-size: 0.85rem;
    color: var(--color-uni-red);
  }


  .share-spinner {
    width: 0.9rem;
    height: 0.9rem;
    border-radius: 999px;
    border: 2px solid rgba(52, 74, 154, 0.25);
    border-top-color: var(--color-uni-blue);
    animation: spin 0.8s linear infinite;
  }

  .share-error {
    font-size: 0.85rem;
    color: var(--color-uni-red);
  }

  @keyframes spin {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }

  .cea-section + .cea-section {
    margin-top: var(--spacing-md);
  }

  .cea-heading {
    margin: 0 0 var(--spacing-xs);
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--color-uni-dark-blue);
  }

  .cea-annotations {
    border: 1px solid rgba(52, 74, 154, 0.2);
    border-radius: var(--radius-sm);
    overflow-x: auto;
    overflow-y: hidden;
  }

  .cea-annotations table {
    width: 100%;
    min-width: 540px;
    border-collapse: collapse;
    font-size: 0.88rem;
  }

  .cea-annotations thead {
    background: rgba(52, 74, 154, 0.08);
    text-align: left;
  }

  .cea-annotations th,
  .cea-annotations td {
    padding: 0.55rem 0.75rem;
    border-bottom: 1px solid rgba(52, 74, 154, 0.12);
    vertical-align: top;
  }

  .cea-annotations tbody tr:last-child td {
    border-bottom: none;
  }

  .cea-annotations a {
    color: var(--color-uni-blue);
    text-decoration: none;
  }

  .cea-annotations a:hover {
    text-decoration: underline;
  }

  .cea-annotations code {
    font-size: 0.82rem;
    padding: 0.1rem 0.25rem;
    background: rgba(52, 74, 154, 0.08);
    border-radius: var(--radius-xs);
  }

  @media (max-width: 640px) {
    .cea-annotations table {
      font-size: 0.82rem;
    }

    .cea-annotations th,
    .cea-annotations td {
      padding: 0.5rem;
    }
  }

</style>
