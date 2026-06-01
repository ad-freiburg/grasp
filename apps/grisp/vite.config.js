import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig({
	plugins: [sveltekit()],
	define: {
		__API_BASE__: JSON.stringify(process.env.API_BASE || '/api'),
		__COPYRIGHT__: JSON.stringify(process.env.COPYRIGHT || 'University of Freiburg'),
		__MATERIAL_URL__: JSON.stringify(process.env.MATERIAL_URL || 'grisp')
	}
});
