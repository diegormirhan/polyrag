<script lang="ts">
	import type { Span } from '$lib/api';

	let { spans }: { spans: Span[] } = $props();

	// Only the newest trace. Showing every span the backend ever emitted would be
	// a log, not a timeline — the question is "what happened for THIS answer".
	let trace = $derived.by(() => {
		if (spans.length === 0) return [];
		const newest = spans.reduce((a, b) => (b.start_ms > a.start_ms ? b : a));
		return spans
			.filter((s) => s.trace_id === newest.trace_id)
			.sort((a, b) => a.start_ms - b.start_ms);
	});

	let origin = $derived(trace.length ? trace[0].start_ms : 0);
	let span_ms = $derived(
		trace.length ? Math.max(...trace.map((s) => s.start_ms + s.duration_ms)) - origin : 1
	);

	// Depth by walking parent links, so nesting shows without a recursive component.
	let depth = $derived.by(() => {
		const byId = new Map(trace.map((s) => [s.span_id, s]));
		const levels = new Map<string, number>();
		for (const s of trace) {
			let level = 0;
			let parent = s.parent_span_id;
			while (parent && byId.has(parent) && level < 8) {
				level += 1;
				parent = byId.get(parent)!.parent_span_id;
			}
			levels.set(s.span_id, level);
		}
		return levels;
	});

	function shorten(name: string) {
		return name.replace(/^pipeline\.|^llama\./, '').replace(/^POST \/api\/v1\//, '');
	}
</script>

<section>
	<h3>Pipeline</h3>

	{#if trace.length === 0}
		<p class="idle">Steps appear here as the pipeline runs.</p>
	{:else}
		<ol>
			{#each trace as span (span.span_id)}
				<li style:padding-left="calc({depth.get(span.span_id) ?? 0} * 0.625rem)">
					<span class="name" title={span.name}>{shorten(span.name)}</span>
					<span class="track">
						<span
							class="fill"
							class:leaf={!span.name.startsWith('pipeline')}
							style:margin-left="{((span.start_ms - origin) / span_ms) * 100}%"
							style:width="max(2px, {(span.duration_ms / span_ms) * 100}%)"
						></span>
					</span>
					<span class="tabular value">{Math.round(span.duration_ms)}<i>ms</i></span>
				</li>
			{/each}
		</ol>
	{/if}
</section>

<style>
	section {
		padding: var(--space-4) var(--space-5);
		border-bottom: 1px solid var(--separator);
	}

	h3 {
		color: var(--text-tertiary);
		text-transform: uppercase;
		font-size: var(--text-xs);
		font-weight: 590;
		letter-spacing: 0.06em;
		margin-bottom: var(--space-3);
	}

	.idle {
		color: var(--text-secondary);
		font-size: var(--text-md);
	}

	ol {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 0.375rem;
	}

	li {
		display: grid;
		grid-template-columns: 7.5rem 1fr 4rem;
		align-items: center;
		gap: var(--space-2);
		font-size: var(--text-xs);
		color: var(--text-secondary);
	}

	.name {
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.track {
		height: 4px;
		border-radius: 2px;
		background: var(--surface-sunken);
	}

	.fill {
		display: block;
		height: 100%;
		border-radius: 2px;
		background: var(--accent);
	}

	/* Model and network calls in neutral, pipeline stages in accent: at a glance
	   you see how much of the bar is our code and how much is waiting on a model. */
	.fill.leaf {
		background: var(--separator-strong);
	}

	.value {
		text-align: right;
		color: var(--text-tertiary);
	}

	.value i {
		font-style: normal;
		opacity: 0.55;
		margin-left: 1px;
	}
</style>
