<script>
  import { STATUS } from '../../utils/provenance.js';

  // an item with its status, heading the popups of query items and of the
  // later uses of items
  export let item;
</script>

<div class="head">
  <strong>{item.label ? `${item.label} (${item.written})` : item.written}</strong>
  {#if item.status}
    <span class={`badge badge--${item.status}`}>{STATUS[item.status]?.text ?? item.status}</span>
  {/if}
</div>

<style>
  .head {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.4rem;
    overflow-wrap: anywhere;
    color: var(--color-uni-dark-blue);
  }

  .badge {
    border-radius: 999px;
    padding: 0.05rem 0.5rem;
    font-size: 0.72rem;
    font-weight: 600;
  }

  .badge--retrieved {
    background: color-mix(in srgb, var(--color-uni-green) 16%, transparent);
    color: #006b57;
  }

  .badge--verified {
    background: color-mix(in srgb, var(--color-uni-yellow) 22%, transparent);
    color: #6b5d12;
  }

  .badge--noted {
    background: color-mix(in srgb, var(--color-uni-blue) 14%, transparent);
    color: var(--color-uni-blue);
  }

  .badge--unverified {
    background: color-mix(in srgb, var(--color-uni-red) 14%, transparent);
    color: var(--color-uni-red);
  }
</style>
