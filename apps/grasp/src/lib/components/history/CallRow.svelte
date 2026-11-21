<script>
  // a call that showed or used an item, in the popups of query items and of
  // the later uses of items; clicking it jumps to the call
  export let what;
  export let tool;
  // shortened arguments as key-value pairs
  export let args = [];
</script>

<button type="button" class="call" on:click>
  <span class="call__what">{what}</span>
  <span class="call__fn">{tool}</span>
  {#each args as [key, value]}
    <span class="call__arg"><span class="call__key">{key}</span> {value}</span>
  {/each}
</button>

<style>
  .call {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.3rem;
    width: 100%;
    padding: 0.25rem 0.35rem;
    border: none;
    border-radius: var(--radius-sm);
    background: none;
    text-align: left;
    font: inherit;
    font-size: 0.75rem;
    color: var(--text-subtle);
    cursor: pointer;
  }

  .call:hover,
  .call:focus-visible {
    background: rgba(52, 74, 154, 0.08);
    outline: none;
  }

  .call__what {
    font-weight: 600;
    color: var(--color-uni-blue);
  }

  .call__what::after {
    content: ' →';
  }

  .call__fn,
  .call__arg {
    padding: 0.05rem 0.4rem;
    border-radius: 999px;
    border: 1px solid rgba(190, 170, 60, 0.45);
    background: #fff;
    overflow-wrap: anywhere;
  }

  .call__fn {
    font-weight: 600;
    color: var(--text-primary);
  }

  .call__key {
    font-weight: 700;
    color: var(--color-uni-yellow);
  }
</style>
