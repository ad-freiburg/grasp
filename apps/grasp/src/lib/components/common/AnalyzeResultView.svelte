<script>
  import { prettyJson } from '../../utils/formatters.js';

  export let payload = null;
  export let modelName = '';
  export let raw = '';

  const MAX_ENTITY_PROPERTIES = 2;
  const MAX_VISIBLE_TEXT = 8;
  const MAX_AUDIO_ITEMS = 8;

  $: entities = Array.isArray(payload?.entities) ? payload.entities : [];
  $: relations = Array.isArray(payload?.relations) ? payload.relations : [];
  $: textVisible = Array.isArray(payload?.text_visible) ? payload.text_visible : [];
  $: audioKeyPoints = Array.isArray(payload?.key_points) ? payload.key_points : [];
  $: audioNoises = Array.isArray(payload?.notable_noises) ? payload.notable_noises : [];
  $: audioIdentities = Array.isArray(payload?.identities) ? payload.identities : [];
  $: isImagePayload =
    entities.length > 0 ||
    relations.length > 0 ||
    textVisible.length > 0 ||
    Boolean(payload?.image_type) ||
    Boolean(payload?.scene_description);
  $: isAudioPayload =
    audioKeyPoints.length > 0 ||
    audioNoises.length > 0 ||
    audioIdentities.length > 0 ||
    Boolean(payload?.summary) ||
    Boolean(payload?.language) ||
    Boolean(payload?.audio_quality);
  $: visibleAudioKeyPoints = audioKeyPoints.slice(0, MAX_AUDIO_ITEMS);
  $: visibleAudioNoises = audioNoises.slice(0, MAX_AUDIO_ITEMS);
  $: visibleAudioIdentities = audioIdentities.slice(0, MAX_AUDIO_ITEMS);
  $: analyzeTitle = isAudioPayload
    ? 'Audio analysis'
    : toText(payload?.image_type) || 'Analyze output';
  $: analyzeSubtitle = isAudioPayload ? toText(payload?.summary) : toText(payload?.scene_description);
  $: entityLabelById = Object.fromEntries(
    entities
      .filter((entity) => entity && typeof entity.id === 'string')
      .map((entity) => [entity.id, entity.label || entity.id])
  );
  $: visibleTextPreview = textVisible.slice(0, MAX_VISIBLE_TEXT);

  function toText(value) {
    if (value === null || value === undefined) return '';
    if (typeof value === 'string') return value;
    if (typeof value === 'number' || typeof value === 'boolean') return String(value);
    try {
      return JSON.stringify(value);
    } catch {
      return String(value);
    }
  }

  function relationSubject(relation) {
    return entityLabelById[relation?.subject_id] || toText(relation?.subject_id) || 'Unknown';
  }

  function relationObject(relation) {
    return entityLabelById[relation?.object_id] || toText(relation?.object_id) || 'Unknown';
  }

  function identityText(entity) {
    const hypothesis = entity?.identity_hypothesis;
    if (!hypothesis || typeof hypothesis !== 'object') return '';
    if (!hypothesis.name) return '';
    const confidence = toText(hypothesis.confidence);
    return confidence ? `${hypothesis.name} (${confidence})` : hypothesis.name;
  }

  function audioIdentityText(identity) {
    if (!identity || typeof identity !== 'object') return '';
    const name = toText(identity?.name) || 'Unknown';
    const confidence = toText(identity?.confidence);
    if (!confidence) return name;
    return `${name} (${confidence})`;
  }

  function rawJson() {
    if (!raw) return '';
    try {
      const parsed = JSON.parse(raw);
      return prettyJson(parsed);
    } catch {
      return raw;
    }
  }
</script>

<section class="analyze">
  <header class="analyze__header">
    <div>
      <h4>{analyzeTitle}</h4>
      {#if analyzeSubtitle}
        <p>{analyzeSubtitle}</p>
      {/if}
    </div>
    {#if modelName}
      <span class="model-chip">{modelName}</span>
    {/if}
  </header>

  {#if isAudioPayload}
    <div class="stats">
      <span>{audioKeyPoints.length} key points</span>
      <span>{audioNoises.length} noises</span>
      <span>{audioIdentities.length} identities</span>
      {#if payload?.language}
        <span>language: {toText(payload.language)}</span>
      {/if}
    </div>
  {:else}
    <div class="stats">
      <span>{entities.length} entities</span>
      <span>{relations.length} relations</span>
      <span>{textVisible.length} text items</span>
    </div>
  {/if}

  {#if isImagePayload && entities.length > 0}
    <div class="entities" aria-label="Entities">
      {#each entities as entity, index (entity?.id ?? index)}
        <article class="entity-card">
          <div class="entity-card__top">
            <h5>{toText(entity?.label) || 'Unnamed entity'}</h5>
            {#if entity?.category}
              <span class="category">{entity.category}</span>
            {/if}
          </div>
          <p class="entity-meta">
            {toText(entity?.locality?.position) || 'unknown position'}
          </p>

          {#if identityText(entity)}
            <p class="entity-identity">{identityText(entity)}</p>
          {/if}

          {#if Array.isArray(entity?.properties) && entity.properties.length > 0}
            <ul class="properties">
              {#each entity.properties.slice(0, MAX_ENTITY_PROPERTIES) as property, propertyIndex (propertyIndex)}
                <li>
                  <span>{toText(property?.name) || 'property'}</span>
                  <span>{toText(property?.value)}</span>
                </li>
              {/each}
            </ul>
          {/if}
        </article>
      {/each}
    </div>
  {/if}

  {#if isImagePayload && relations.length > 0}
    <section class="relations" aria-label="Relations">
      {#each relations as relation, index (index)}
        <p>{relationSubject(relation)} {toText(relation?.predicate) || 'related to'} {relationObject(relation)}</p>
      {/each}
    </section>
  {/if}

  {#if isImagePayload && textVisible.length > 0}
    <details class="text-visible">
      <summary>Visible text</summary>
      <ul>
        {#each visibleTextPreview as item, index (index)}
          <li>{toText(item?.text)}</li>
        {/each}
      </ul>
      {#if textVisible.length > MAX_VISIBLE_TEXT}
        <p class="more">+{textVisible.length - MAX_VISIBLE_TEXT} more items</p>
      {/if}
    </details>
  {/if}

  {#if isAudioPayload}
    {#if payload?.audio_quality}
      <p class="audio-quality"><strong>Audio quality:</strong> {toText(payload.audio_quality)}</p>
    {/if}

    {#if visibleAudioKeyPoints.length > 0}
      <details class="text-visible" open>
        <summary>Key points</summary>
        <ul>
          {#each visibleAudioKeyPoints as item, index (index)}
            <li>{toText(item)}</li>
          {/each}
        </ul>
        {#if audioKeyPoints.length > MAX_AUDIO_ITEMS}
          <p class="more">+{audioKeyPoints.length - MAX_AUDIO_ITEMS} more items</p>
        {/if}
      </details>
    {/if}

    {#if visibleAudioNoises.length > 0}
      <details class="text-visible">
        <summary>Notable noises</summary>
        <ul>
          {#each visibleAudioNoises as item, index (index)}
            <li>{toText(item)}</li>
          {/each}
        </ul>
        {#if audioNoises.length > MAX_AUDIO_ITEMS}
          <p class="more">+{audioNoises.length - MAX_AUDIO_ITEMS} more items</p>
        {/if}
      </details>
    {/if}

    {#if visibleAudioIdentities.length > 0}
      <section class="relations" aria-label="Identities">
        <h5>Identities</h5>
        {#each visibleAudioIdentities as identity, index (index)}
          <p>
            {audioIdentityText(identity)}
            {#if identity?.entity_type}
              - {toText(identity.entity_type)}
            {/if}
            {#if identity?.basis}
              - {toText(identity.basis)}
            {/if}
          </p>
        {/each}
        {#if audioIdentities.length > MAX_AUDIO_ITEMS}
          <p class="more">+{audioIdentities.length - MAX_AUDIO_ITEMS} more items</p>
        {/if}
      </section>
    {/if}
  {/if}

  {#if raw}
    <details class="raw-json">
      <summary>Raw JSON</summary>
      <pre>{rawJson()}</pre>
    </details>
  {/if}
</section>

<style>
  .analyze {
    display: grid;
    gap: var(--spacing-md);
    padding: var(--spacing-md);
    border: 1px solid rgba(52, 74, 154, 0.15);
    border-radius: var(--radius-md);
    background: linear-gradient(180deg, rgba(52, 74, 154, 0.04), rgba(255, 255, 255, 0.9));
  }

  .analyze__header {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: var(--spacing-sm);
  }

  .analyze__header h4 {
    margin: 0;
    font-size: 0.95rem;
    color: var(--color-uni-dark-blue);
  }

  .analyze__header p {
    margin: 0.25rem 0 0;
    color: var(--text-subtle);
    font-size: 0.88rem;
  }

  .model-chip {
    border: 1px solid rgba(52, 74, 154, 0.22);
    border-radius: 999px;
    padding: 0.15rem 0.55rem;
    font-size: 0.72rem;
    color: var(--color-uni-blue);
    background: rgba(52, 74, 154, 0.08);
    white-space: nowrap;
  }

  .stats {
    display: flex;
    flex-wrap: wrap;
    gap: var(--spacing-sm);
  }

  .stats span {
    font-size: 0.78rem;
    border-radius: 999px;
    padding: 0.2rem 0.6rem;
    background: #fff;
    border: 1px solid rgba(0, 0, 0, 0.08);
    color: var(--text-subtle);
  }

  .entities {
    display: grid;
    gap: var(--spacing-sm);
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
  }

  .entity-card {
    border: 1px solid rgba(52, 74, 154, 0.12);
    border-radius: var(--radius-sm);
    background: rgba(255, 255, 255, 0.92);
    padding: 0.6rem 0.7rem;
    display: grid;
    gap: 0.35rem;
  }

  .entity-card__top {
    display: flex;
    justify-content: space-between;
    gap: var(--spacing-sm);
    align-items: center;
  }

  .entity-card__top h5 {
    margin: 0;
    font-size: 0.84rem;
    color: var(--text-primary);
    min-width: 0;
  }

  .category {
    font-size: 0.7rem;
    color: var(--color-uni-blue);
    text-transform: lowercase;
    border-radius: 999px;
    padding: 0.12rem 0.48rem;
    background: rgba(52, 74, 154, 0.08);
    border: 1px solid rgba(52, 74, 154, 0.2);
    white-space: nowrap;
  }

  .entity-meta,
  .entity-identity {
    margin: 0;
    font-size: 0.78rem;
    color: var(--text-subtle);
  }

  .entity-identity {
    color: var(--text-primary);
    font-weight: 500;
  }

  .properties {
    margin: 0;
    padding: 0;
    list-style: none;
    display: grid;
    gap: 0.25rem;
  }

  .properties li {
    display: flex;
    gap: 0.35rem;
    font-size: 0.76rem;
  }

  .properties li span:first-child {
    color: var(--color-uni-blue);
    font-weight: 600;
  }

  .properties li span:last-child {
    color: var(--text-subtle);
  }

  .relations {
    display: grid;
    gap: 0.25rem;
    font-size: 0.82rem;
    color: var(--text-subtle);
  }

  .relations p {
    margin: 0;
  }

  .relations h5 {
    margin: 0 0 0.2rem;
    font-size: 0.82rem;
    color: var(--text-primary);
  }

  .audio-quality {
    margin: 0;
    font-size: 0.82rem;
    color: var(--text-subtle);
  }

  .text-visible summary,
  .raw-json summary {
    cursor: pointer;
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--color-uni-blue);
  }

  .text-visible ul {
    margin: var(--spacing-sm) 0 0;
    padding: 0 0 0 1rem;
    display: grid;
    gap: 0.25rem;
    font-size: 0.78rem;
    color: var(--text-subtle);
  }

  .more {
    margin: 0.35rem 0 0;
    font-size: 0.75rem;
    color: var(--text-subtle);
  }

  .raw-json pre {
    margin: var(--spacing-sm) 0 0;
    padding: 0.6rem;
    border-radius: var(--radius-sm);
    background: #f6f8fc;
    border: 1px solid rgba(52, 74, 154, 0.12);
    font-size: 0.74rem;
    overflow: auto;
    color: #22315f;
  }
</style>
