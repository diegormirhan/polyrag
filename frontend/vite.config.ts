import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	server: {
		// Vite does not read PORT on its own, so an assigned port would be ignored
		// and 5173 tried anyway — which is taken by another project on this machine.
		port: Number(process.env.PORT) || 5173,
		// Proxied rather than called cross-origin: the browser sees one origin, so
		// there is no CORS config to keep in sync between dev and production. This
		// is also why the dev port is free to move: nothing is pinned to it.
		proxy: {
			'/api': { target: 'http://127.0.0.1:8000', changeOrigin: true, ws: true },
			'/docs': { target: 'http://127.0.0.1:8000', changeOrigin: true },
			'/openapi.json': { target: 'http://127.0.0.1:8000', changeOrigin: true }
		}
	}
});
