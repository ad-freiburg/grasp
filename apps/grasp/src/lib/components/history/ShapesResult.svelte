<script>
  export let result;

  $: shapes = Array.isArray(result?.shapes) ? result.shapes : [];
  // e.g. "Shapes (page 1 of 10+)" as label and page info
  $: summary = (result?.summary ?? '').match(/^(.*?)(?: \((.*)\))?$/);

  // cardinality as a regex quantifier after the targets like in ShEx, none for
  // exactly one value
  const CARDINALITY = {
    '': 'most instances have exactly one value',
    '?': 'some instances have one value, others none',
    '+': 'most instances have one or more values',
    '*': 'some instances have several values, others none'
  };

  function href(iri) {
    return /^https?:\/\//.test(iri ?? '') ? iri : null;
  }

  // label of a shape or property, its identifier if it has none
  function name(item) {
    return item.label ?? item.short_iri;
  }

  function propertyTitle(prop, shape) {
    const id = prop.short_iri;
    const variants = prop.variants?.length ? ` as ${prop.variants.join('/')}` : '';
    const detail = prop.inverse
      ? `instances of these classes point to ${name(shape)} via ${name(prop)}`
      : (CARDINALITY[prop.cardinality ?? ''] ?? '');
    return `${id}${variants}${detail ? `\n${detail}` : ''}`;
  }

  // datatypes without their prefix, e.g. string for xsd:string
  function targetText(target) {
    switch (target.kind) {
      case 'class':
        return target.label ?? target.short_iri ?? target.value;
      case 'literal':
        return target.value.replace(/^[\w-]+:/, '');
      case 'bnode':
        return 'blank node';
      default:
        return 'IRI';
    }
  }

  function targetTitle(target) {
    switch (target.kind) {
      case 'class':
        return target.short_iri ?? target.value;
      case 'literal':
        return `literal of type ${target.value}`;
      case 'bnode':
        return 'blank node';
      default:
        return 'IRI without a known class';
    }
  }

  // properties of a section that are not shown, as a sentence
  function hiddenText(omitted, filtered) {
    const count = (n) => `${n.toLocaleString()} more ${n === 1 ? 'property' : 'properties'}`;
    const parts = [];
    if (omitted) parts.push(`${count(omitted)} not shown`);
    if (filtered) parts.push(`${count(filtered)} hidden because few instances have them`);
    return parts.join(', ');
  }

  function sections(shape) {
    return [
      { name: 'Outgoing', props: outgoing(shape), hidden: hiddenText(shape.omitted, shape.filtered) },
      {
        name: 'Incoming',
        props: incoming(shape),
        hidden: hiddenText(shape.incoming_omitted, shape.incoming_filtered)
      }
    ];
  }

  function noteText(shape) {
    const hidden = shape.omitted + shape.filtered + shape.incoming_omitted + shape.incoming_filtered;
    return shape.properties.length || hidden ? '' : 'No properties found for this class';
  }

  function outgoing(shape) {
    return shape.properties.filter((prop) => !prop.inverse);
  }

  function incoming(shape) {
    return shape.properties.filter((prop) => prop.inverse);
  }
</script>

<div class="shapes-result">
  {#if result?.summary}
    <p class="list-label">
      {summary?.[1] ?? result.summary}
      {#if summary?.[2]}<span class="list-label__meta">· {summary[2]}</span>{/if}
    </p>
  {/if}

  {#each shapes as shape, s}
    <div class="shape" class:shape--ranked={shapes.length > 1}>
      <div class="shape__head" data-path={`shapes.${s}`}>
        {#if shapes.length > 1}<span class="list-row__rank">{s + 1}</span>{/if}
        <strong>{name(shape)}</strong>
        {#if shape.short_iri}
          {#if href(shape.iri)}
            <a href={href(shape.iri)} target="_blank" rel="noopener noreferrer"><code>{shape.short_iri}</code></a>
          {:else}
            <code>{shape.short_iri}</code>
          {/if}
        {/if}
      </div>

      {#each sections(shape) as section}
        {#if section.props.length || section.hidden}
          <p class="list-label section">{section.name}</p>
          <ul class="list-rows">
            {#each section.props as prop}
              {@const p = shape.properties.indexOf(prop)}
              <li class="list-row prop" class:prop--incoming={prop.inverse}>
                <span class="prop__name" data-path={`shapes.${s}.properties.${p}`}>
                  {#if href(prop.iri)}
                    <a class="prop__card" title={propertyTitle(prop, shape)} href={href(prop.iri)} target="_blank" rel="noopener noreferrer">{name(prop)}</a>
                  {:else}
                    <span class="prop__card" title={propertyTitle(prop, shape)}>{name(prop)}</span>
                  {/if}
                </span>
                <span class="targets">
                  {#each prop.targets.length ? prop.targets : [{ value: 'IRI', kind: 'iri' }] as target, t}
                    {@const link = target.kind === 'class' ? href(target.iri) : null}
                    <span data-path={`shapes.${s}.properties.${p}.targets.${t}`}>
                      {#if link}
                        <a class={`target target--${target.kind}`} title={targetTitle(target)} href={link} target="_blank" rel="noopener noreferrer">{targetText(target)}</a>
                      {:else}
                        <span class={`target target--${target.kind}`} title={targetTitle(target)}>{targetText(target)}</span>
                      {/if}
                    </span>
                  {/each}
                  {#if !prop.inverse && prop.cardinality}
                    <span class="quantifier" title={CARDINALITY[prop.cardinality] ?? ''}>{prop.cardinality}</span>
                  {/if}
                </span>
                {#if prop.inverse}
                  <span class="arrow">→ {name(shape)}</span>
                {/if}
              </li>
            {/each}
          </ul>
          {#if section.hidden}
            <p class="hidden">{section.hidden}</p>
          {/if}
        {/if}
      {/each}

      {#if noteText(shape)}
        <p class="hidden">{noteText(shape)}</p>
      {/if}
    </div>
  {/each}
</div>

<style>
  .shapes-result {
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--text-primary);
    overflow-wrap: anywhere;
  }

  .shape + .shape {
    margin-top: var(--spacing-md);
    padding-top: var(--spacing-sm);
    border-top: 1px solid rgba(52, 74, 154, 0.1);
  }

  .shape__head {
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
    margin-bottom: var(--spacing-xs);
  }

  .shape__head .list-row__rank {
    flex: 0 0 1.2rem;
  }

  .section {
    margin-top: var(--spacing-sm);
  }

  .prop {
    display: grid;
    grid-template-columns: minmax(7rem, 13rem) 1fr;
    align-items: start;
    gap: var(--spacing-sm);
  }

  /* incoming: source classes first, then the property and the class,
     flowing like a triple */
  .prop--incoming {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
  }

  .prop--incoming .targets {
    order: -1;
  }

  .prop__card {
    display: inline-block;
    padding: 0.05rem 0.5rem;
    border: 1px solid rgba(0, 0, 0, 0.16);
    border-radius: var(--radius-sm);
    background: #fff;
    color: var(--text-primary);
    font-size: 0.85rem;
    line-height: 1.5;
    text-decoration: none;
  }

  a.prop__card:hover {
    border-color: rgba(52, 74, 154, 0.45);
    text-decoration: underline;
  }

  .targets {
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
  }

  .target {
    padding: 0.05rem 0.45rem;
    border: 1px solid transparent;
    border-radius: var(--radius-sm);
    font-size: 0.82rem;
    line-height: 1.5;
    text-decoration: none;
  }

  a.target:hover {
    text-decoration: underline;
  }

  .target--class {
    color: #23397f;
    background: rgba(52, 74, 154, 0.08);
    border-color: rgba(52, 74, 154, 0.22);
  }

  .target--literal {
    color: #1d5e3b;
    background: rgba(46, 160, 90, 0.08);
    border-color: rgba(46, 160, 90, 0.28);
    font-family: 'Fira Code', 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
    font-size: 0.78rem;
  }

  .target--iri,
  .target--bnode {
    color: var(--text-subtle);
    background: rgba(0, 0, 0, 0.03);
    border-color: rgba(0, 0, 0, 0.12);
    font-size: 0.78rem;
  }

  /* estimated cardinality, boxed apart from the property name */
  .quantifier {
    display: inline-block;
    align-self: center;
    min-width: 1.3rem;
    padding: 0.05rem 0.3rem;
    border: 1px dashed rgba(0, 0, 0, 0.2);
    border-radius: var(--radius-sm);
    background: rgba(0, 0, 0, 0.03);
    color: var(--text-subtle);
    font-family: 'Fira Code', 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.5;
    text-align: center;
    cursor: help;
  }

  .arrow {
    line-height: 1.9;
    font-size: 0.8rem;
    color: var(--text-subtle);
    white-space: nowrap;
  }

  .hidden {
    margin: var(--spacing-xs) 0 0;
    font-size: 0.8rem;
    color: var(--text-subtle);
  }

  code {
    font-family: 'Fira Code', 'SFMono-Regular', Consolas, 'Liberation Mono', monospace;
    font-size: 0.875em;
  }

  a {
    color: var(--color-uni-blue);
  }

  @media (max-width: 640px) {
    .prop {
      grid-template-columns: minmax(0, 1fr);
    }
  }
</style>
