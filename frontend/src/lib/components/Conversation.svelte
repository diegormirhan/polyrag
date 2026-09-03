<script lang="ts">
	import { marked } from 'marked';
	import type { Turn } from '$lib/types';

	let { turns }: { turns: Turn[] } = $props();

	let viewport: HTMLDivElement;
	let pinned = $state(true);

	// Follow the answer as it streams, but stop the moment the reader scrolls up.
	// Yanking someone back to the bottom while they are reading is the rudest
	// thing a chat UI can do.
	function onScroll() {
		const distance = viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight;
		pinned = distance < 48;
	}

	$effect(() => {
		// Touching the last turn's text is what subscribes this effect to streaming.
		turns.at(-1)?.text;
		if (pinned && viewport) viewport.scrollTop = viewport.scrollHeight;
	});
</script>

<div class="viewport" bind:this={viewport} onscroll={onScroll}>
	{#if turns.length === 0}
		<div class="empty">
			<h2>Ask across three stores at once</h2>
			<p>
				A router decides — with arithmetic, not a model — whether your question belongs to the
				SQL tables, the vector index, or the knowledge graph. Watch it decide on the right.
			</p>
		</div>
	{/if}

	{#each turns as turn, index (index)}
		<article class={turn.role}>
			{#if turn.role === 'user'}
				<p>{turn.text}</p>
			{:else}
				{#if turn.cacheHit}
					<span class="badge">from cache</span>
				{/if}
				<div class="prose">{@html marked.parse(turn.text)}</div>
				{#if turn.pending && !turn.text}
					<span class="thinking" aria-label="Working"></span>
				{/if}
			{/if}
		</article>
	{/each}
</div>

<style>
	.viewport {
		flex: 1;
		overflow-y: auto;
		overscroll-behavior: contain;
		padding: var(--space-6) var(--space-5) var(--space-4);
		display: flex;
		flex-direction: column;
		gap: var(--space-5);
	}

	.empty {
		margin: auto 0;
		max-width: 34rem;
		align-self: center;
		text-align: center;
	}

	.empty h2 {
		font-size: var(--text-xl);
		letter-spacing: -0.019em;
		margin-bottom: var(--space-3);
	}

	.empty p {
		color: var(--text-secondary);
		font-size: var(--text-base);
		line-height: 1.6;
	}

	article {
		max-width: 44rem;
		width: 100%;
		margin: 0 auto;
		animation: rise var(--duration-enter) var(--ease-out) both;
	}

	/* A short rise, not a slide across the screen: enough to register as new,
	   not enough to make the reader wait for it. */
	@keyframes rise {
		from {
			opacity: 0;
			transform: translateY(6px);
		}
	}

	.user p {
		background: var(--accent);
		color: var(--on-accent);
		padding: var(--space-3) var(--space-4);
		border-radius: var(--radius-lg);
		border-bottom-right-radius: var(--radius-sm);
		max-width: 32rem;
		margin-left: auto;
		width: fit-content;
	}

	.badge {
		display: inline-block;
		font-size: var(--text-xs);
		color: var(--accent);
		background: var(--accent-soft);
		border-radius: 999px;
		padding: 0.125rem 0.5rem;
		margin-bottom: var(--space-2);
	}

	.prose :global(p) {
		margin: 0 0 var(--space-3);
	}

	.prose :global(p:last-child) {
		margin-bottom: 0;
	}

	.prose :global(strong) {
		font-weight: 590;
	}

	.prose :global(table) {
		border-collapse: collapse;
		width: 100%;
		font-size: var(--text-md);
		margin: var(--space-3) 0;
	}

	.prose :global(th),
	.prose :global(td) {
		text-align: left;
		padding: var(--space-2) var(--space-3);
		border-bottom: 1px solid var(--separator);
	}

	.prose :global(th) {
		color: var(--text-tertiary);
		font-weight: 510;
		font-size: var(--text-xs);
		text-transform: uppercase;
		letter-spacing: 0.06em;
	}

	.prose :global(td) {
		font-variant-numeric: tabular-nums;
	}

	/* One pulsing dot rather than three bouncing ones: it says "working" without
	   performing impatience. */
	.thinking {
		display: inline-block;
		width: 7px;
		height: 7px;
		border-radius: 50%;
		background: var(--text-tertiary);
		animation: breathe 1.4s var(--ease-standard) infinite;
	}

	@keyframes breathe {
		0%,
		100% {
			opacity: 0.25;
		}
		50% {
			opacity: 0.85;
		}
	}
</style>
