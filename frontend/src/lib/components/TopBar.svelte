<script lang="ts">
	import { page } from '$app/state';
	import { api, type ServiceHealth } from '$lib/api';
	import { setTheme, theme, type Theme } from '$lib/theme.svelte';

	const NAV = [
		{ href: '/', label: 'Chat' },
		{ href: '/corpus', label: 'Corpus' },
		{ href: '/telemetry', label: 'Telemetry' }
	];

	const THEMES: { value: Theme; label: string; path: string }[] = [
		{ value: 'system', label: 'Match the system', path: 'M8 2a6 6 0 000 12z' },
		{ value: 'light', label: 'Light', path: 'M8 4.5a3.5 3.5 0 100 7 3.5 3.5 0 000-7zM8 1v1.6M8 13.4V15M15 8h-1.6M2.6 8H1M12.9 3.1l-1.1 1.1M4.2 11.8l-1.1 1.1M12.9 12.9l-1.1-1.1M4.2 4.2L3.1 3.1' },
		{ value: 'dark', label: 'Dark', path: 'M13 9.8A5.8 5.8 0 016.2 3a5.8 5.8 0 106.8 6.8z' }
	];

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

	function describe(service: ServiceHealth) {
		if (!service.up) return 'offline';
		return service.sleeping ? 'asleep — VRAM released' : 'ready';
	}
</script>

<header>
	<a class="brand" href="/">PolyRAG</a>

	<nav>
		{#each NAV as item (item.href)}
			<a href={item.href} aria-current={page.url.pathname === item.href ? 'page' : undefined}>
				{item.label}
			</a>
		{/each}
	</nav>

	<div class="right">
		<div class="services">
			{#if reachable}
				{#each services as service (service.name)}
					<span
						class="service"
						class:down={!service.up}
						class:asleep={service.sleeping}
						title="{service.name} — {describe(service)} ({service.url})"
					>
						<i aria-hidden="true"></i>{service.name}
					</span>
				{/each}
			{:else}
				<span class="service down"><i aria-hidden="true"></i>backend offline</span>
			{/if}
		</div>

		<!-- Three explicit states, not a cycling button: a toggle that hides which
		     option is active makes the user click to find out. -->
		<div class="themes" role="group" aria-label="Colour theme">
			{#each THEMES as option (option.value)}
				<button
					class:active={theme.value === option.value}
					aria-pressed={theme.value === option.value}
					title={option.label}
					onclick={() => setTheme(option.value)}
				>
					<svg viewBox="0 0 16 16" width="13" height="13" aria-hidden="true">
						<path
							d={option.path}
							fill={option.value === 'light' ? 'none' : 'currentColor'}
							stroke="currentColor"
							stroke-width={option.value === 'light' ? 1.3 : 0}
							stroke-linecap="round"
						/>
					</svg>
					<span class="sr">{option.label}</span>
				</button>
			{/each}
		</div>

		<a class="api" href="/docs" target="_blank" rel="noreferrer">API</a>
	</div>
</header>

<style>
	header {
		display: flex;
		align-items: center;
		gap: var(--space-5);
		height: var(--topbar-height);
		padding: 0 var(--space-5);
		/* Translucent chrome with content scrolling under it, rather than an opaque
		   strip that permanently costs its own height. */
		background: var(--chrome);
		backdrop-filter: blur(20px) saturate(180%);
		-webkit-backdrop-filter: blur(20px) saturate(180%);
		border-bottom: 1px solid var(--separator);
		position: sticky;
		top: 0;
		z-index: 20;
	}

	.brand {
		font-size: var(--text-lg);
		font-weight: 590;
		letter-spacing: -0.014em;
		color: inherit;
		text-decoration: none;
		flex-shrink: 0;
	}

	nav {
		display: flex;
		gap: var(--space-1);
	}

	nav a {
		color: var(--text-secondary);
		text-decoration: none;
		font-size: var(--text-md);
		padding: 0.25rem 0.625rem;
		border-radius: var(--radius-sm);
		transition:
			color var(--duration-press) var(--ease-standard),
			background var(--duration-press) var(--ease-standard);
	}

	nav a:hover {
		color: var(--text-primary);
	}

	nav a[aria-current='page'] {
		color: var(--text-primary);
		background: var(--surface-sunken);
		font-weight: 510;
	}

	.right {
		display: flex;
		align-items: center;
		gap: var(--space-4);
		margin-left: auto;
	}

	.services {
		display: flex;
		align-items: center;
		gap: var(--space-3);
		font-size: var(--text-xs);
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
	   carries the state, so it survives greyscale and colour blindness. */
	.service i {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--positive);
	}

	.service.asleep i {
		background: transparent;
		border: 1px solid var(--text-tertiary);
	}

	.service.down i {
		background: var(--warning);
	}

	.service.down {
		color: var(--warning);
	}

	.themes {
		display: flex;
		gap: 2px;
		padding: 2px;
		border-radius: 999px;
		background: var(--surface-sunken);
	}

	.themes button {
		width: 1.75rem;
		height: 1.75rem;
		display: grid;
		place-items: center;
		border-radius: 999px;
		color: var(--text-tertiary);
		transition:
			background var(--duration-press) var(--ease-standard),
			color var(--duration-press) var(--ease-standard);
	}

	.themes button:hover {
		color: var(--text-secondary);
	}

	.themes button.active {
		background: var(--surface);
		color: var(--text-primary);
		box-shadow: 0 1px 2px rgb(0 0 0 / 0.08);
	}

	.api {
		color: var(--accent);
		text-decoration: none;
		font-size: var(--text-xs);
		font-weight: 510;
		padding: 0.125rem 0.5rem;
		border-radius: 999px;
		background: var(--accent-soft);
	}

	@media (max-width: 900px) {
		.services {
			display: none;
		}
	}
</style>
