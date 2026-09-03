<script lang="ts">
	import { api, type ServiceHealth } from '$lib/api';

	let services = $state<ServiceHealth[]>([]);
	let reachable = $state(true);

	async function poll() {
		try {
			services = (await api.health()).services;
			reachable = true;
		} catch {
			reachable = false;
		}
	}

	$effect(() => {
		poll();
		// Slow on purpose. A health probe is not free, and nothing here changes
		// second to second — a fast poll would be motion for its own sake.
		const timer = setInterval(poll, 10_000);
		return () => clearInterval(timer);
	});

	function label(service: ServiceHealth) {
		if (!service.up) return 'offline';
		return service.sleeping ? 'asleep — VRAM released' : 'ready';
	}
</script>

<header>
	<div class="brand">
		<h1>PolyRAG</h1>
		<span>Federated retrieval with deterministic routing</span>
	</div>

	<div class="services">
		{#if reachable}
			{#each services as service (service.name)}
				<span
					class="service"
					class:down={!service.up}
					class:asleep={service.sleeping}
					title="{service.url} — {label(service)}"
				>
					<i aria-hidden="true"></i>{service.name}
				</span>
			{/each}
		{:else}
			<span class="service down"><i aria-hidden="true"></i>backend offline</span>
		{/if}
		<a href="/docs" target="_blank" rel="noreferrer">API</a>
	</div>
</header>

<style>
	header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: var(--space-5);
		padding: var(--space-3) var(--space-5);
		/* Translucent chrome with content scrolling under it, rather than an opaque
		   strip that permanently costs its own height. */
		background: var(--chrome);
		backdrop-filter: blur(20px) saturate(180%);
		-webkit-backdrop-filter: blur(20px) saturate(180%);
		border-bottom: 1px solid var(--separator);
	}

	.brand {
		display: flex;
		align-items: baseline;
		gap: var(--space-3);
		min-width: 0;
	}

	.brand h1 {
		font-size: 1.0625rem;
		letter-spacing: -0.014em;
	}

	.brand span {
		color: var(--text-tertiary);
		font-size: 0.75rem;
		letter-spacing: 0.004em;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.services {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		font-size: 0.6875rem;
		letter-spacing: 0.006em;
		color: var(--text-secondary);
	}

	.service {
		display: inline-flex;
		align-items: center;
		gap: 0.375rem;
		white-space: nowrap;
	}

	/* Filled dot = loaded. Hollow = asleep but reachable. Amber = down. Shape
	   carries the state, so it survives greyscale and colour-blindness. */
	.service i {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--positive);
		box-shadow: 0 0 0 1px color-mix(in srgb, var(--positive) 30%, transparent);
	}

	.service.asleep i {
		background: transparent;
		border: 1px solid var(--text-tertiary);
		box-shadow: none;
	}

	.service.down i {
		background: var(--warning);
		box-shadow: none;
	}

	.service.down {
		color: var(--warning);
	}

	a {
		color: var(--accent);
		text-decoration: none;
		font-weight: 510;
		padding: 0.125rem 0.5rem;
		border-radius: 999px;
		background: var(--accent-soft);
		transition: opacity var(--duration-press) var(--ease-standard);
	}

	a:active {
		opacity: 0.6;
	}

	@media (max-width: 720px) {
		.brand span,
		.service {
			display: none;
		}
	}
</style>
