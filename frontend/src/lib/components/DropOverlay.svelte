<script lang="ts">
	let { onfiles }: { onfiles: (files: File[]) => void } = $props();

	// dragenter/dragleave fire for every child element the pointer crosses. Counting
	// them is what keeps the overlay from flickering as the cursor moves inside it.
	let depth = $state(0);
	let active = $derived(depth > 0);

	function hasFiles(event: DragEvent) {
		return event.dataTransfer?.types.includes('Files') ?? false;
	}

	function onEnter(event: DragEvent) {
		if (hasFiles(event)) depth += 1;
	}

	function onLeave() {
		depth = Math.max(0, depth - 1);
	}

	function onOver(event: DragEvent) {
		if (hasFiles(event)) event.preventDefault();
	}

	function onDrop(event: DragEvent) {
		depth = 0;
		if (!hasFiles(event)) return;
		event.preventDefault();
		const files = Array.from(event.dataTransfer?.files ?? []);
		if (files.length) onfiles(files);
	}
</script>

<svelte:window
	ondragenter={onEnter}
	ondragleave={onLeave}
	ondragover={onOver}
	ondrop={onDrop}
/>

{#if active}
	<div class="scrim" aria-hidden="true">
		<div class="plate">
			<strong>Drop to ingest</strong>
			<span>The router decides where it belongs — a table, a passage, or a rule.</span>
		</div>
	</div>
{/if}

<style>
	/* The whole window is the target. A permanent dashed rectangle would occupy
	   the layout forever to serve an action taken once in a while; the interface
	   should answer the drag, not anticipate it. */
	.scrim {
		position: fixed;
		inset: 0;
		z-index: 40;
		display: grid;
		place-items: center;
		background: color-mix(in srgb, var(--bg) 55%, transparent);
		backdrop-filter: blur(12px) saturate(160%);
		-webkit-backdrop-filter: blur(12px) saturate(160%);
		animation: materialize var(--duration-enter) var(--ease-out) both;
		pointer-events: none;
	}

	.plate {
		display: grid;
		gap: var(--space-2);
		justify-items: center;
		text-align: center;
		padding: var(--space-6) var(--space-7);
		border-radius: var(--radius-lg);
		background: var(--surface);
		border: 1px solid var(--separator);
		box-shadow: var(--shadow-overlay);
		max-width: 26rem;
	}

	.plate strong {
		font-size: 1.0625rem;
		font-weight: 590;
		letter-spacing: -0.014em;
	}

	.plate span {
		font-size: 0.8125rem;
		color: var(--text-secondary);
		line-height: 1.5;
	}

	/* Blur and scale together, so the surface reads as a material arriving rather
	   than a rectangle fading in. */
	@keyframes materialize {
		from {
			opacity: 0;
			backdrop-filter: blur(0) saturate(100%);
			-webkit-backdrop-filter: blur(0) saturate(100%);
		}
	}

	@media (prefers-reduced-motion: reduce) {
		.scrim {
			animation: none;
		}
	}
</style>
