<script lang="ts">
	import { api, streamAnswer, watchSpans, type RouteDecision, type Source, type Span } from '$lib/api';
	import Composer from '$lib/components/Composer.svelte';
	import Conversation from '$lib/components/Conversation.svelte';
	import DropOverlay from '$lib/components/DropOverlay.svelte';
	import RouteDecisionCard from '$lib/components/RouteDecisionCard.svelte';
	import SourcesList from '$lib/components/SourcesList.svelte';
	import SpanTimeline from '$lib/components/SpanTimeline.svelte';
	import StatusBar from '$lib/components/StatusBar.svelte';
	import type { Turn } from '$lib/types';

	let turns = $state<Turn[]>([]);
	let decision = $state<RouteDecision | null>(null);
	let sources = $state<Source[]>([]);
	let cacheHit = $state(false);
	let busy = $state(false);
	let spans = $state<Span[]>([]);
	let notice = $state<string | null>(null);

	$effect(() => watchSpans((span) => (spans = [...spans.slice(-400), span])));

	async function ask(message: string) {
		busy = true;
		decision = null;
		sources = [];
		cacheHit = false;
		spans = [];
		turns = [...turns, { role: 'user', text: message }, { role: 'assistant', text: '', pending: true }];
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

	async function ingest(files: File[]) {
		for (const file of files) {
			notice = `Ingesting ${file.name}…`;
			try {
				const reports = await api.ingest(file);
				const chunks = reports.reduce((total, report) => total + report.chunks, 0);
				const routes = [...new Set(reports.flatMap((report) => report.routes))];
				notice = `${file.name} — ${chunks} chunk${chunks === 1 ? '' : 's'} into ${routes.join(', ')}`;
			} catch {
				notice = `${file.name} could not be ingested.`;
			}
		}
		setTimeout(() => (notice = null), 6000);
	}
</script>

<StatusBar />
<DropOverlay onfiles={ingest} />

<main>
	<section class="chat">
		<Conversation {turns} />
		{#if notice}
			<p class="notice" role="status">{notice}</p>
		{/if}
		<footer><Composer {busy} onsubmit={ask} onattach={(file) => ingest([file])} /></footer>
	</section>

	<aside>
		<RouteDecisionCard {decision} {cacheHit} />
		<SpanTimeline {spans} />
		<SourcesList {sources} />
	</aside>
</main>

<style>
	main {
		display: grid;
		grid-template-columns: minmax(0, 1fr) 22rem;
		height: calc(100dvh - 3.25rem);
	}

	.chat {
		display: flex;
		flex-direction: column;
		min-width: 0;
	}

	footer {
		padding: var(--space-3) var(--space-5) var(--space-5);
	}

	.notice {
		max-width: 44rem;
		width: 100%;
		margin: 0 auto;
		padding: 0 var(--space-5) var(--space-2);
		font-size: 0.75rem;
		color: var(--text-secondary);
		animation: rise var(--duration-enter) var(--ease-out) both;
	}

	@keyframes rise {
		from {
			opacity: 0;
			transform: translateY(4px);
		}
	}

	aside {
		border-left: 1px solid var(--separator);
		background: var(--surface);
		overflow-y: auto;
		overscroll-behavior: contain;
	}

	/* Below this the panel stops being a companion and starts being a competitor
	   for the reading column, so it moves under the conversation instead. */
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
