<script lang="ts">
	import { api, type ServerState } from '$lib/api';

	let servers = $state<ServerState[]>([]);
	let failed = $state<string | null>(null);
	/** Names with a start/stop in flight, so their button cannot be clicked twice. */
	let busy = $state(new Set<string>());
	let note = $state<string | null>(null);

	const ABOUT: Record<string, string> = {
		llm: 'Writes SQL, extracts entities, breaks routing ties and composes every answer.',
		ocr: 'Reads text and tables out of images. Releases its VRAM after a minute idle.',
		embeddings: 'Turns text into the 1024-dimension vectors the router and the cache compare.',
		judge: 'Resolves the router’s gray zone when geometry alone is not confident.',
		qdrant: 'Stores the vectors for the vectorial route and searches them with HNSW.'
	};

	async function load() {
		try {
			servers = await api.servers();
			failed = null;
		} catch {
			failed = 'The backend is not reachable. Is uvicorn running on port 8000?';
		}
	}

	$effect(() => {
		load();
		// Faster than the top bar's poll, because this is the page where the user is
		// waiting for a state to change: a model takes tens of seconds to load, and
		// a ten-second poll would make a working start look stuck.
		const timer = setInterval(load, 2000);
		return () => clearInterval(timer);
	});

	async function act(name: string, action: 'start' | 'stop') {
		busy = new Set(busy).add(name);
		try {
			const outcome = action === 'start' ? await api.startServer(name) : await api.stopServer(name);
			note =
				outcome.result === 'unmanaged'
					? `${name} is running but was not started here, so it has no pid to stop. Close it where you started it.`
					: null;
			await load();
		} catch (error) {
			failed = error instanceof Error ? error.message : `${name} could not be ${action}ed.`;
		} finally {
			const next = new Set(busy);
			next.delete(name);
			busy = next;
		}
	}

	async function all(action: 'start' | 'stop') {
		// In parallel: models load independently, and starting them one at a time
		// would serialise a wait that the GPU handles concurrently anyway.
		await Promise.all(
			servers
				.filter((server) => (action === 'start' ? !server.up : server.up))
				.map((server) => act(server.name, action))
		);
	}

	function describe(server: ServerState) {
		if (!server.up) return 'offline';
		return server.sleeping ? 'asleep — VRAM released' : 'ready';
	}

	const anyDown = $derived(servers.some((server) => !server.up));
	const anyUp = $derived(servers.some((server) => server.up));
</script>

<main>
	<header>
		<h1>Servers</h1>
		<p>
			The models and the vector store, each its own process. They run without a window; their
			output goes to <code>data/run/&lt;name&gt;.log</code>. Stopping one hands its VRAM back
			immediately — useful on a 16&nbsp;GB card when something else needs the GPU.
		</p>
	</header>

	<div class="bulk">
		<button class="primary" disabled={!anyDown} onclick={() => all('start')}>Start all</button>
		<button disabled={!anyUp} onclick={() => all('stop')}>Stop all</button>
	</div>

	{#if failed}<p class="failed">{failed}</p>{/if}
	{#if note}<p class="note">{note}</p>{/if}

	<ul class="list">
		{#each servers as server (server.name)}
			<li class="card" class:down={!server.up}>
				<div class="head">
					<span class="dot" class:asleep={server.sleeping} class:off={!server.up}></span>
					<strong>{server.name}</strong>
					<span class="label">{server.label}</span>
					<span class="state">{describe(server)}</span>
				</div>

				<p class="about">{ABOUT[server.name] ?? ''}</p>

				<div class="foot">
					<a href={server.url} target="_blank" rel="noreferrer">{server.url}</a>
					{#if server.up && !server.managed}
						<span class="unmanaged" title="Started outside the app — no pid to stop safely">
							external
						</span>
					{/if}
					<button
						class:primary={!server.up}
						disabled={busy.has(server.name) || (server.up && !server.managed)}
						onclick={() => act(server.name, server.up ? 'stop' : 'start')}
					>
						{#if busy.has(server.name)}
							working…
						{:else}
							{server.up ? 'Stop' : 'Start'}
						{/if}
					</button>
				</div>
			</li>
		{/each}
	</ul>
</main>

<style>
	main {
		max-width: 60rem;
		margin: 0 auto;
		padding: var(--space-6) var(--space-5) var(--space-7);
		display: grid;
		gap: var(--space-5);
	}

	header h1 {
		margin-bottom: var(--space-2);
	}

	header p {
		max-width: 44rem;
		color: var(--text-secondary);
		font-size: var(--text-base);
		line-height: 1.6;
	}

	code {
		font-family: var(--font-mono);
		font-size: 0.9em;
	}

	.bulk {
		display: flex;
		gap: var(--space-3);
	}

	.failed {
		color: var(--warning);
		font-size: var(--text-md);
	}

	.note {
		color: var(--text-secondary);
		font-size: var(--text-md);
	}

	.list {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(19rem, 1fr));
		gap: var(--space-4);
		list-style: none;
	}

	.card {
		display: grid;
		gap: var(--space-3);
		padding: var(--space-4);
		border-radius: var(--radius-lg);
		border: 1px solid var(--separator);
		background: var(--surface);
	}

	.head {
		display: flex;
		align-items: baseline;
		gap: var(--space-2);
	}

	.head strong {
		font-size: var(--text-base);
		font-weight: 590;
	}

	.label {
		font-size: var(--text-xs);
		color: var(--text-tertiary);
	}

	.state {
		margin-left: auto;
		font-size: var(--text-xs);
		color: var(--text-secondary);
		white-space: nowrap;
	}

	/* Filled = loaded, hollow = asleep, amber = down. Shape carries the state, so
	   it survives greyscale and colour blindness — same rule as the top bar. */
	.dot {
		width: 7px;
		height: 7px;
		border-radius: 50%;
		background: var(--positive);
		align-self: center;
	}

	.dot.asleep {
		background: transparent;
		border: 1px solid var(--text-tertiary);
	}

	.dot.off {
		background: var(--warning);
	}

	.about {
		font-size: var(--text-md);
		color: var(--text-secondary);
		line-height: 1.5;
	}

	.foot {
		display: flex;
		align-items: center;
		gap: var(--space-3);
	}

	.foot a {
		font-family: var(--font-mono);
		font-size: var(--text-xs);
		color: var(--text-tertiary);
		text-decoration: none;
	}

	.foot a:hover {
		color: var(--text-secondary);
	}

	.unmanaged {
		font-size: var(--text-xs);
		color: var(--text-tertiary);
		padding: 0.0625rem 0.375rem;
		border-radius: 999px;
		background: var(--surface-sunken);
	}

	button {
		margin-left: auto;
		font-size: var(--text-md);
		font-weight: 510;
		padding: 0.3125rem 0.875rem;
		border-radius: var(--radius-sm);
		background: var(--surface-sunken);
		color: var(--text-primary);
		white-space: nowrap;
		transition:
			background var(--duration-press) var(--ease-standard),
			opacity var(--duration-press) var(--ease-standard);
	}

	.bulk button {
		margin-left: 0;
	}

	button.primary {
		background: var(--accent);
		color: var(--on-accent);
	}

	button:disabled {
		opacity: 0.4;
		cursor: default;
	}

	button:not(:disabled):hover {
		filter: brightness(0.96);
	}
</style>
