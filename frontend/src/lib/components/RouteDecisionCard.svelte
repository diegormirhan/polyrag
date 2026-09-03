<script lang="ts">
	import type { RouteDecision } from '$lib/api';

	let { decision, cacheHit = false }: { decision: RouteDecision | null; cacheHit?: boolean } =
		$props();

	const STAGE_COPY: Record<string, string> = {
		heuristic: 'Decided by the tabularity heuristic — no embedding needed.',
		embedding: 'Decided by cosine similarity and margin. No model was asked.',
		llm_judge: 'Too close to call on the numbers — a model broke the tie.'
	};

	// Sorted every render so the winner is always first; the ordering is the
	// ranking, not an arbitrary object key order.
	let ranked = $derived(
		decision ? Object.entries(decision.scores).sort(([, a], [, b]) => b - a) : []
	);
</script>

<section>
	<h3>Route</h3>

	{#if cacheHit}
		<p class="cached">
			Answered from the semantic cache. No routing, no retrieval, no model call.
		</p>
	{:else if !decision}
		<p class="idle">Waiting for a question.</p>
	{:else}
		<div class="chosen">
			<strong>{decision.route}</strong>
			<span class="stage" class:judged={decision.decision_stage === 'llm_judge'}>
				{decision.decision_stage === 'llm_judge' ? 'model tiebreak' : decision.decision_stage}
			</span>
		</div>

		<ol>
			{#each ranked as [route, score], index (route)}
				<li class:won={index === 0}>
					<span class="name">{route}</span>
					<span class="track"><span class="fill" style:width="{score * 100}%"></span></span>
					<span class="tabular value">{score.toFixed(3)}</span>
				</li>
			{/each}
		</ol>

		<dl>
			<dt>margin</dt>
			<dd class="tabular">{decision.margin.toFixed(3)}</dd>
		</dl>

		<p class="explain">{STAGE_COPY[decision.decision_stage]}</p>
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
		font-size: 0.6875rem;
		font-weight: 590;
		letter-spacing: 0.06em;
		margin-bottom: var(--space-3);
	}

	.idle,
	.cached {
		color: var(--text-secondary);
		font-size: 0.8125rem;
	}

	.chosen {
		display: flex;
		align-items: center;
		gap: var(--space-2);
		margin-bottom: var(--space-3);
	}

	.chosen strong {
		font-size: 1.125rem;
		font-weight: 590;
		letter-spacing: -0.016em;
	}

	.stage {
		font-size: 0.6875rem;
		color: var(--text-secondary);
		background: var(--surface-sunken);
		border-radius: 999px;
		padding: 0.125rem 0.5rem;
	}

	/* The tiebreak is the one case where a model, not arithmetic, made the call.
	   It is worth marking — that is the honest version of the pitch. */
	.stage.judged {
		color: var(--warning);
		background: color-mix(in srgb, var(--warning) 12%, transparent);
	}

	ol {
		list-style: none;
		margin: 0 0 var(--space-3);
		padding: 0;
		display: grid;
		gap: var(--space-2);
	}

	li {
		display: grid;
		grid-template-columns: 4.75rem 1fr 2.75rem;
		align-items: center;
		gap: var(--space-2);
		font-size: 0.75rem;
		color: var(--text-secondary);
	}

	li.won {
		color: var(--text-primary);
	}

	.track {
		height: 4px;
		border-radius: 2px;
		background: var(--surface-sunken);
		overflow: hidden;
	}

	/* Width is the score itself, on a 0–1 scale — not normalised to the leader.
	   Rescaling would make a weak win look decisive. */
	.fill {
		display: block;
		height: 100%;
		border-radius: 2px;
		background: var(--separator-strong);
		transition: width var(--duration-large) var(--ease-out);
	}

	li.won .fill {
		background: var(--accent);
	}

	.value {
		text-align: right;
		font-size: 0.6875rem;
	}

	dl {
		display: flex;
		align-items: baseline;
		gap: var(--space-2);
		margin: 0 0 var(--space-2);
	}

	dt {
		font-size: 0.6875rem;
		color: var(--text-tertiary);
		text-transform: uppercase;
		letter-spacing: 0.06em;
	}

	dd {
		margin: 0;
		font-size: 0.8125rem;
		font-weight: 510;
	}

	.explain {
		font-size: 0.75rem;
		color: var(--text-tertiary);
		line-height: 1.45;
	}
</style>
