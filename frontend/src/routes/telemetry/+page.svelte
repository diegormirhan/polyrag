<script lang="ts">
	import type { Span } from '$lib/api';
	import SpanTimeline from '$lib/components/SpanTimeline.svelte';
	import { clearSpans, spans, traces } from '$lib/spans.svelte';

	let selectedId = $state<string | null>(null);

	let all = $derived(traces(spans.items));
	// Falls back to the newest trace so the page is never blank while spans arrive,
	// but an explicit selection wins and is not stolen by the next request.
	let selected = $derived(all.find((trace) => trace.id === selectedId) ?? all[0] ?? null);

	function attributes(trace: { spans: Span[] } | null) {
		if (!trace) return [];
		const merged = new Map<string, string | number | boolean>();
		for (const span of trace.spans) {
			for (const [key, value] of Object.entries(span.attributes)) merged.set(key, value);
		}
		return [...merged].sort(([a], [b]) => a.localeCompare(b));
	}

	function when(ms: number) {
		return new Date(ms).toLocaleTimeString(undefined, { hour12: false });
	}
</script>

<main>
	<header>
		<div>
			<h1>Telemetry</h1>
			<p>
				Every step the backend takes emits an OpenTelemetry span. These arrive over a WebSocket
				straight from the SDK — no collector, no Jaeger, no extra process.
			</p>
		</div>
		<button onclick={clearSpans} disabled={spans.items.length === 0}>Clear</button>
	</header>

	{#if all.length === 0}
		<p class="empty">
			No traces yet. Ask a question or ingest a file and the pipeline will report itself here.
		</p>
	{:else}
		<div class="layout">
			<ol class="traces">
				{#each all as trace (trace.id)}
					<li>
						<button
							class:active={selected?.id === trace.id}
							onclick={() => (selectedId = trace.id)}
						>
							<span class="name">{trace.name}</span>
							<span class="meta">
								<span class="tabular">{Math.round(trace.duration)}ms</span>
								<span>{trace.spans.length} spans</span>
								<span class="tabular time">{when(trace.start)}</span>
							</span>
						</button>
					</li>
				{/each}
			</ol>

			<section class="detail">
				<SpanTimeline spans={selected?.spans ?? []} />

				<div class="attributes">
					<h3 class="eyebrow">Attributes</h3>
					{#if attributes(selected).length === 0}
						<p class="empty">This trace carries no attributes.</p>
					{:else}
						<dl>
							{#each attributes(selected) as [key, value] (key)}
								<div>
									<dt>{key}</dt>
									<dd class:number={typeof value === 'number'}>{value}</dd>
								</div>
							{/each}
						</dl>
					{/if}
				</div>
			</section>
		</div>
	{/if}
</main>

<style>
	main {
		max-width: 72rem;
		margin: 0 auto;
		padding: var(--space-6) var(--space-5) var(--space-7);
		display: grid;
		gap: var(--space-5);
	}

	header {
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
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

	header button {
		flex-shrink: 0;
		font-size: var(--text-sm);
		color: var(--text-secondary);
		padding: 0.3rem 0.75rem;
		border-radius: 999px;
		border: 1px solid var(--separator);
		transition: background var(--duration-press) var(--ease-standard);
	}

	header button:hover:not(:disabled) {
		background: var(--surface-sunken);
	}

	header button:disabled {
		color: var(--text-tertiary);
		cursor: default;
	}

	.layout {
		display: grid;
		grid-template-columns: 20rem minmax(0, 1fr);
		gap: var(--space-5);
		align-items: start;
	}

	.traces {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 2px;
		max-height: 70dvh;
		overflow-y: auto;
	}

	.traces button {
		width: 100%;
		text-align: left;
		display: grid;
		gap: 2px;
		padding: var(--space-2) var(--space-3);
		border-radius: var(--radius-sm);
		transition: background var(--duration-press) var(--ease-standard);
	}

	.traces button:hover {
		background: var(--surface-sunken);
	}

	.traces button.active {
		background: var(--accent-soft);
	}

	.traces .name {
		font-size: var(--text-md);
		display: block;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.traces button.active .name {
		color: var(--accent);
		font-weight: 510;
	}

	.meta {
		display: flex;
		gap: var(--space-2);
		font-size: var(--text-xs);
		color: var(--text-tertiary);
	}

	.time {
		margin-left: auto;
	}

	.detail {
		border: 1px solid var(--separator);
		border-radius: var(--radius-md);
		background: var(--surface);
		overflow: hidden;
	}

	.attributes {
		padding: var(--space-4) var(--space-5);
	}

	dl {
		margin: 0;
		display: grid;
		gap: 2px;
	}

	dl div {
		display: grid;
		grid-template-columns: 16rem minmax(0, 1fr);
		gap: var(--space-3);
		font-size: var(--text-sm);
		padding: 0.25rem 0;
		border-bottom: 1px solid var(--separator);
	}

	dt {
		color: var(--text-tertiary);
		font-family: var(--font-mono);
		overflow: hidden;
		text-overflow: ellipsis;
	}

	dd {
		margin: 0;
		color: var(--text-primary);
		overflow-wrap: anywhere;
	}

	dd.number {
		font-family: var(--font-mono);
		font-variant-numeric: tabular-nums;
	}

	.empty {
		font-size: var(--text-md);
		color: var(--text-tertiary);
	}

	@media (max-width: 900px) {
		.layout {
			grid-template-columns: minmax(0, 1fr);
		}

		.traces {
			max-height: 14rem;
		}
	}
</style>
