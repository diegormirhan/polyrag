<script lang="ts">
	import { api, type Corpus, type Route } from '$lib/api';

	let corpus = $state<Corpus | null>(null);
	let failed = $state(false);
	let dragging = $state(false);
	let picker: HTMLInputElement;
	let busy = $state<string | null>(null);

	const ROUTE_COPY: Record<Route, string> = {
		relational: 'Rows and columns, queried with generated SQL.',
		vectorial: 'Free text, retrieved by cosine similarity.',
		graph: 'Entities and relations, ranked by Personalized PageRank.'
	};

	async function load() {
		try {
			corpus = await api.corpus();
			failed = false;
		} catch {
			failed = true;
		}
	}

	$effect(() => {
		load();
	});

	async function send(files: File[]) {
		for (const file of files) {
			busy = file.name;
			try {
				await api.ingest(file);
			} catch {
				failed = true;
			}
		}
		busy = null;
		await load();
	}

	function onDrop(event: DragEvent) {
		dragging = false;
		const files = Array.from(event.dataTransfer?.files ?? []);
		if (files.length) send(files);
	}

	function when(seconds: number) {
		return new Date(seconds * 1000).toLocaleString(undefined, {
			dateStyle: 'medium',
			timeStyle: 'short'
		});
	}

	function size(bytes: number) {
		if (bytes < 1024) return `${bytes} B`;
		if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
		return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
	}

	/** How many chunks of a file went to each route, in a stable order. */
	function tally(routes: Route[]) {
		const order: Route[] = ['relational', 'vectorial', 'graph'];
		const counts = new Map<Route, number>();
		for (const route of routes) counts.set(route, (counts.get(route) ?? 0) + 1);
		return order.filter((route) => counts.has(route)).map((route) => ({ route, count: counts.get(route)! }));
	}
</script>

<main>
	<header>
		<h1>Corpus</h1>
		<p>
			Everything that has been ingested, and where the router sent it. A file becomes chunks, and
			each chunk is routed on its own — which is why one file can land in more than one store.
		</p>
	</header>

	<!-- A visible target belongs here and nowhere else: this is the page about
	     ingestion, so the drop area is the subject rather than an interruption. -->
	<button
		class="dropzone"
		class:dragging
		ondragenter={() => (dragging = true)}
		ondragleave={() => (dragging = false)}
		ondragover={(event) => event.preventDefault()}
		ondrop={onDrop}
		onclick={() => picker.click()}
	>
		{#if busy}
			<strong>Ingesting {busy}…</strong>
			<span>OCR, chunking, routing and storage all run before this returns.</span>
		{:else}
			<strong>Drop files here, or click to choose</strong>
			<span>Images and PDFs, spreadsheets, plain text and Markdown.</span>
		{/if}
	</button>
	<input
		bind:this={picker}
		type="file"
		multiple
		hidden
		accept=".png,.jpg,.jpeg,.webp,.bmp,.txt,.md,.csv,.xlsx,.pdf,.docx,.pptx"
		onchange={(event) => {
			const files = Array.from((event.currentTarget as HTMLInputElement).files ?? []);
			(event.currentTarget as HTMLInputElement).value = '';
			if (files.length) send(files);
		}}
	/>

	{#if failed}
		<p class="failed">The backend is not reachable. Is uvicorn running on port 8000?</p>
	{/if}

	<section class="stores">
		{#each ['relational', 'vectorial', 'graph'] as const as route (route)}
			<article>
				<h2>{route}</h2>
				<p class="what">{ROUTE_COPY[route]}</p>
				{#if route === 'relational' && corpus?.stores.relational}
					{#if corpus.stores.relational.tables.length}
						<ul class="figures">
							{#each corpus.stores.relational.tables as table (table.name)}
								<li><span>{table.name}</span><b class="tabular">{table.rows}</b></li>
							{/each}
						</ul>
					{:else}
						<p class="empty">No tables yet.</p>
					{/if}
				{:else if route === 'vectorial' && corpus?.stores.vectorial}
					<ul class="figures">
						<li><span>points</span><b class="tabular">{corpus.stores.vectorial.points}</b></li>
						<li><span>collection</span><b>{corpus.stores.vectorial.collection}</b></li>
					</ul>
				{:else if route === 'graph' && corpus?.stores.graph}
					<ul class="figures">
						<li><span>entities</span><b class="tabular">{corpus.stores.graph.entities}</b></li>
						<li><span>relations</span><b class="tabular">{corpus.stores.graph.relations}</b></li>
						<li><span>chunks</span><b class="tabular">{corpus.stores.graph.chunks}</b></li>
					</ul>
				{/if}
			</article>
		{/each}
	</section>

	<section class="files">
		<h3 class="eyebrow">Ingested files <span>{corpus?.files.length ?? ''}</span></h3>
		{#if corpus && corpus.files.length === 0}
			<p class="empty">Nothing ingested yet.</p>
		{:else if corpus}
			<table>
				<thead>
					<tr>
						<th>File</th>
						<th>Chunks</th>
						<th>Stored in</th>
						<th>Size</th>
						<th>Ingested</th>
					</tr>
				</thead>
				<tbody>
					{#each corpus.files as file (file.name)}
						<tr>
							<td class="name">{file.name}</td>
							<td class="tabular">{file.chunks || '—'}</td>
							<td>
								{#if file.routes.length}
									<span class="pills">
										{#each tally(file.routes) as entry (entry.route)}
											<span class="pill">{entry.route}<b>{entry.count}</b></span>
										{/each}
									</span>
								{:else}
									<span class="unknown" title="Ingested before this was recorded">—</span>
								{/if}
							</td>
							<td class="tabular">{size(file.size_bytes)}</td>
							<td class="when">{when(file.ingested_at)}</td>
						</tr>
					{/each}
				</tbody>
			</table>
		{/if}
	</section>
</main>

<style>
	main {
		max-width: 60rem;
		margin: 0 auto;
		padding: var(--space-6) var(--space-5) var(--space-7);
		display: grid;
		gap: var(--space-6);
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

	.dropzone {
		display: grid;
		gap: var(--space-1);
		justify-items: center;
		text-align: center;
		width: 100%;
		padding: var(--space-6);
		border-radius: var(--radius-lg);
		border: 1px dashed var(--separator-strong);
		background: var(--surface);
		transition:
			border-color var(--duration-press) var(--ease-standard),
			background var(--duration-press) var(--ease-standard);
	}

	.dropzone.dragging {
		border-color: var(--accent);
		background: var(--accent-soft);
	}

	.dropzone strong {
		font-size: var(--text-base);
		font-weight: 590;
	}

	.dropzone span {
		font-size: var(--text-md);
		color: var(--text-secondary);
	}

	.failed {
		color: var(--warning);
		font-size: var(--text-md);
	}

	.stores {
		display: grid;
		grid-template-columns: repeat(auto-fit, minmax(15rem, 1fr));
		gap: var(--space-4);
	}

	.stores article {
		padding: var(--space-4);
		border-radius: var(--radius-md);
		background: var(--surface);
		border: 1px solid var(--separator);
	}

	.stores h2 {
		font-size: var(--text-base);
		margin-bottom: var(--space-1);
	}

	.what {
		font-size: var(--text-sm);
		color: var(--text-tertiary);
		line-height: 1.45;
		margin-bottom: var(--space-3);
	}

	.figures {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: var(--space-1);
	}

	.figures li {
		display: flex;
		justify-content: space-between;
		gap: var(--space-3);
		font-size: var(--text-sm);
		color: var(--text-secondary);
		padding: 0.125rem 0;
		border-bottom: 1px solid var(--separator);
	}

	.figures b {
		font-weight: 510;
		color: var(--text-primary);
	}

	.empty {
		font-size: var(--text-md);
		color: var(--text-tertiary);
	}

	table {
		width: 100%;
		border-collapse: collapse;
		font-size: var(--text-md);
	}

	th {
		text-align: left;
		padding: 0 var(--space-3) var(--space-2);
		color: var(--text-tertiary);
		font-size: var(--text-xs);
		font-weight: 590;
		text-transform: uppercase;
		letter-spacing: 0.06em;
		border-bottom: 1px solid var(--separator);
	}

	td {
		padding: var(--space-3);
		border-bottom: 1px solid var(--separator);
		color: var(--text-secondary);
		vertical-align: middle;
	}

	td.name {
		color: var(--text-primary);
		font-weight: 510;
	}

	.when {
		white-space: nowrap;
		color: var(--text-tertiary);
	}

	.pills {
		display: inline-flex;
		flex-wrap: wrap;
		gap: var(--space-1);
	}

	.pill {
		display: inline-flex;
		align-items: center;
		gap: 0.3rem;
		font-size: var(--text-xs);
		padding: 0.125rem 0.5rem;
		border-radius: 999px;
		background: var(--surface-sunken);
		color: var(--text-secondary);
	}

	.pill b {
		font-weight: 590;
		color: var(--text-primary);
		font-variant-numeric: tabular-nums;
	}

	.unknown {
		color: var(--text-tertiary);
	}
</style>
