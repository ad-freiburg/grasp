<script>
  import { createEventDispatcher } from 'svelte';
  import KgItem from './KgItem.svelte';

  const dispatch = createEventDispatcher();

  // hovering an item dispatches itemhover with its positions in the query,
  // null when the pointer leaves it
  function hover(item) {
    dispatch('itemhover', { spans: item?.spans?.length ? item.spans : null });
  }

  // the items of a query per type, e.g. its entities and properties
  export let groups = [];
</script>

<div class="selection-groups">
  {#each groups as group}
    <p class="list-label">{group.name}</p>
    <ul class="list-rows">
      {#each group.items as item}
        <li
          class="list-row"
          class:row--in-query={item.spans?.length}
          on:pointerenter={() => hover(item)}
          on:pointerleave={() => hover(null)}
        >
          <div class="list-row__body"><KgItem {item} /></div>
        </li>
      {/each}
    </ul>
  {/each}
</div>

<style>
  .row--in-query:hover {
    background: rgba(52, 74, 154, 0.04);
  }

  .selection-groups {
    font-size: 0.95rem;
    line-height: 1.6;
    color: var(--text-primary);
    overflow-wrap: anywhere;
  }
</style>
