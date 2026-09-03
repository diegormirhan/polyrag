<script lang="ts">
	import '../app.css';
	import { api } from '$lib/api';
	import DropOverlay from '$lib/components/DropOverlay.svelte';
	import TopBar from '$lib/components/TopBar.svelte';
	import { connectSpans } from '$lib/spans.svelte';

	let { children } = $props();
	let notice = $state<string | null>(null);

	// One telemetry socket for the whole app, opened where the app is, not where
	// a particular view happens to be mounted.
	$effect(() => connectSpans());

	// Dropping a file works from any view: the ingestion pipeline does not care
	// which page you were looking at, and neither should the interface.
	async function ingest(files: File[]) {
		for (const file of files) {
			notice = `Ingesting ${file.name}…`;
			try {
				const reports = await api.ingest(file);
				const chunks = reports.reduce((total, report) => total + report.chunks, 0);
				const routes = [...new Set(reports.flatMap((report) => report.routes))];
				notice = `${file.name} — ${chunks} chunk${chunks === 1 ? '' : 's'} into ${routes.join(', ')}`;
			} catch (error) {
				notice = error instanceof Error ? error.message : `${file.name} could not be ingested.`;
			}
		}
		setTimeout(() => (notice = null), 6000);
	}
</script>

<TopBar />
<DropOverlay onfiles={ingest} />

{@render children()}

{#if notice}
	<p class="toast" role="status">{notice}</p>
{/if}

<style>
	.toast {
		position: fixed;
		bottom: var(--space-5);
		left: 50%;
		transform: translateX(-50%);
		z-index: 30;
		padding: var(--space-2) var(--space-4);
		border-radius: 999px;
		font-size: var(--text-sm);
		color: var(--text-primary);
		background: var(--chrome);
		backdrop-filter: blur(20px) saturate(180%);
		-webkit-backdrop-filter: blur(20px) saturate(180%);
		border: 1px solid var(--separator);
		box-shadow: var(--shadow-overlay);
		animation: rise var(--duration-enter) var(--ease-out) both;
	}

	@keyframes rise {
		from {
			opacity: 0;
			transform: translate(-50%, 8px);
		}
	}
</style>
