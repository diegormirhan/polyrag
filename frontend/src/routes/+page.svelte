<script lang="ts">
	import { streamAnswer, type RouteDecision, type Source } from '$lib/api';
	import Composer from '$lib/components/Composer.svelte';
	import Conversation from '$lib/components/Conversation.svelte';
	import RouteDecisionCard from '$lib/components/RouteDecisionCard.svelte';
	import SourcesList from '$lib/components/SourcesList.svelte';
	import SpanTimeline from '$lib/components/SpanTimeline.svelte';
	import { clearSpans, spans } from '$lib/spans.svelte';
	import type { Turn } from '$lib/types';

	let turns = $state<Turn[]>([]);
	let decision = $state<RouteDecision | null>(null);
	let sources = $state<Source[]>([]);
	let cacheHit = $state(false);
	let busy = $state(false);

	async function ask(message: string) {
		busy = true;
		decision = null;
		sources = [];
		cacheHit = false;
		// The panel answers "what happened for THIS question", so the buffer starts
		// empty. The full history stays available under Telemetry.
		clearSpans();
		turns = [
			...turns,
			{ role: 'user', text: message },
			{ role: 'assistant', text: '', pending: true }
		];
		const reply = turns.length - 1;

		try {
			for await (const event of streamAnswer(message)) {
				switch (event.type) {
					case 'cache_hit':
						cacheHit = true;
						turns[reply].cacheHit = true;
						break;
					case 'decision':
						decision = event;
						break;
					case 'sources':
						sources = event.sources;
						break;
					case 'token':
						turns[reply].text += event.text;
						break;
				}
			}
		} catch {
			turns[reply].text = 'The backend is not reachable. Is uvicorn running on port 8000?';
		} finally {
			turns[reply].pending = false;
			busy = false;
		}
	}
</script>

<main>
	<section class="chat">
		<Conversation {turns} />
		<footer><Composer {busy} onsubmit={ask} /></footer>
	</section>

	<aside>
		<RouteDecisionCard {decision} {cacheHit} />
		<SpanTimeline spans={spans.items} />
		<SourcesList {sources} />
	</aside>
</main>

<style>
	main {
		display: grid;
		grid-template-columns: minmax(0, 1fr) 25rem;
		height: calc(100dvh - var(--topbar-height));
	}

	.chat {
		display: flex;
		flex-direction: column;
		min-width: 0;
	}

	footer {
		padding: var(--space-3) var(--space-5) var(--space-5);
	}

	aside {
		border-left: 1px solid var(--separator);
		background: var(--surface);
		overflow-y: auto;
		overscroll-behavior: contain;
	}

	/* Below this the panel stops being a companion and starts competing with the
	   reading column, so it moves under the conversation instead. */
	@media (max-width: 900px) {
		main {
			grid-template-columns: minmax(0, 1fr);
			grid-template-rows: minmax(0, 1fr) auto;
		}

		aside {
			border-left: none;
			border-top: 1px solid var(--separator);
			max-height: 45dvh;
		}
	}
</style>
