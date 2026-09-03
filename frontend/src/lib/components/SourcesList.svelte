<script lang="ts">
	import type { Source } from '$lib/api';

	let { sources }: { sources: Source[] } = $props();

	// Relational results are SQL rows (arbitrary columns); the other two RAGs
	// return {text, score}. One check tells them apart without a type tag.
	function isChunk(source: Source): source is { text: string; score: number } {
		return typeof source.text === 'string';
	}

	function columns(source: Source) {
		return Object.entries(source).map(([key, value]) => `${key}: ${value}`).join(' · ');
	}
</script>

<section>
	<h3>Sources <span class="count">{sources.length || ''}</span></h3>

	{#if sources.length === 0}
		<p class="idle">Retrieved passages will be listed here.</p>
	{:else}
		<ul>
			{#each sources as source, index (index)}
				<li>
					{#if isChunk(source)}
						<span class="tabular score">{source.score.toFixed(3)}</span>
						<p>{source.text}</p>
					{:else}
						<span class="tabular score">row</span>
						<p class="row">{columns(source)}</p>
					{/if}
				</li>
			{/each}
		</ul>
	{/if}
</section>

<style>
	section {
		padding: var(--space-4) var(--space-5);
	}

	h3 {
		display: flex;
		align-items: baseline;
		gap: var(--space-2);
		color: var(--text-tertiary);
		text-transform: uppercase;
		font-size: 0.6875rem;
		font-weight: 590;
		letter-spacing: 0.06em;
		margin-bottom: var(--space-3);
	}

	.count {
		letter-spacing: 0;
	}

	.idle {
		color: var(--text-secondary);
		font-size: 0.8125rem;
	}

	ul {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: var(--space-3);
	}

	li {
		display: grid;
		grid-template-columns: 2.75rem 1fr;
		gap: var(--space-2);
		align-items: start;
	}

	.score {
		font-size: 0.6875rem;
		color: var(--text-tertiary);
		padding-top: 0.125rem;
	}

	p {
		font-size: 0.75rem;
		line-height: 1.5;
		color: var(--text-secondary);
		/* Clamped rather than scrolled: the panel answers "what came back", and the
		   full text belongs in the answer, not here. */
		display: -webkit-box;
		-webkit-line-clamp: 3;
		line-clamp: 3;
		-webkit-box-orient: vertical;
		overflow: hidden;
	}

	p.row {
		font-family: var(--font-mono);
		font-size: 0.6875rem;
		color: var(--text-primary);
	}
</style>
