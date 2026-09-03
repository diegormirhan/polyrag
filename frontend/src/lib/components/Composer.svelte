<script lang="ts">
	let {
		busy = false,
		onsubmit,
		onattach
	}: { busy?: boolean; onsubmit: (message: string) => void; onattach: (file: File) => void } =
		$props();

	let value = $state('');
	let field: HTMLTextAreaElement;
	let picker: HTMLInputElement;

	function send() {
		const message = value.trim();
		if (!message || busy) return;
		value = '';
		resize();
		onsubmit(message);
	}

	// Enter sends, Shift+Enter breaks the line — the convention people already
	// have for this exact control.
	function onKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			send();
		}
	}

	function resize() {
		field.style.height = 'auto';
		field.style.height = `${Math.min(field.scrollHeight, 200)}px`;
	}

	function pick(event: Event) {
		const file = (event.target as HTMLInputElement).files?.[0];
		if (file) onattach(file);
		picker.value = '';
	}
</script>

<div class="composer">
	<button class="attach" onclick={() => picker.click()} title="Add a file to the corpus">
		<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
			<path
				d="M8 3.5v9M3.5 8h9"
				stroke="currentColor"
				stroke-width="1.5"
				stroke-linecap="round"
				fill="none"
			/>
		</svg>
		<span class="sr">Add a file</span>
	</button>
	<input
		bind:this={picker}
		type="file"
		hidden
		onchange={pick}
		accept=".png,.jpg,.jpeg,.webp,.bmp,.txt,.md,.csv,.xlsx,.pdf,.docx,.pptx"
	/>

	<textarea
		bind:this={field}
		bind:value
		rows="1"
		placeholder="Ask about the tables, the documents, or the rules…"
		oninput={resize}
		onkeydown={onKeydown}
	></textarea>

	<button class="send" onclick={send} disabled={busy || !value.trim()} title="Send">
		<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
			<path
				d="M8 13V3.5M4 7l4-3.5L12 7"
				stroke="currentColor"
				stroke-width="1.6"
				stroke-linecap="round"
				stroke-linejoin="round"
				fill="none"
			/>
		</svg>
		<span class="sr">Send</span>
	</button>
</div>

<style>
	.composer {
		display: flex;
		align-items: flex-end;
		gap: var(--space-2);
		max-width: 44rem;
		width: 100%;
		margin: 0 auto;
		padding: var(--space-2);
		background: var(--surface);
		border: 1px solid var(--separator);
		border-radius: var(--radius-lg);
		box-shadow: var(--shadow-raised);
		transition: border-color var(--duration-press) var(--ease-standard);
	}

	.composer:focus-within {
		border-color: color-mix(in srgb, var(--accent) 45%, transparent);
	}

	textarea {
		flex: 1;
		resize: none;
		border: none;
		background: none;
		color: inherit;
		font: inherit;
		padding: 0.4375rem 0.25rem;
		max-height: 200px;
		outline: none;
	}

	textarea::placeholder {
		color: var(--text-tertiary);
	}

	button {
		flex-shrink: 0;
		width: 2rem;
		height: 2rem;
		display: grid;
		place-items: center;
		border-radius: 50%;
		/* Feedback on press, not on release — waiting for the click to fire before
		   acknowledging the touch is what makes an interface feel dead. */
		transition:
			transform var(--duration-press) var(--ease-standard),
			background var(--duration-press) var(--ease-standard),
			opacity var(--duration-press) var(--ease-standard);
	}

	button:active:not(:disabled) {
		transform: scale(0.92);
	}

	.attach {
		color: var(--text-secondary);
	}

	.attach:hover {
		background: var(--surface-sunken);
	}

	.send {
		background: var(--accent);
		color: #fff;
	}

	.send:disabled {
		background: var(--surface-sunken);
		color: var(--text-tertiary);
		cursor: default;
	}

	.sr {
		position: absolute;
		width: 1px;
		height: 1px;
		overflow: hidden;
		clip-path: inset(50%);
		white-space: nowrap;
	}
</style>
