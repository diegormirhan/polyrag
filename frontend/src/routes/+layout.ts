// Every page here reads live state from the backend, so there is nothing that
// exists before it answers: nothing to prerender, and no server to render on.
export const ssr = false;
export const prerender = false;
