# MCTX Visualization Frontend

This directory contains the Next.js frontend application for MCTX visualization, designed to be deployed on Vercel.

## Features

- Interactive tree visualization of MCTS search results
- Real-time metrics dashboard
- Control panel for simulation parameters
- Responsive design for desktop and mobile
- Dark and light mode support

## Prerequisites

- Node.js 16+ and npm
- Vercel account (for deployment)
- MCTX API backend running (see `/deployment/fastapi`)

## Local Development

1. Install dependencies:

```bash
npm install
```

2. Set up environment variables:

Create a `.env.local` file with:

```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

3. Start the development server:

```bash
npm run dev
```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Deployment to Vercel

### Using Vercel CLI

1. Install Vercel CLI:

```bash
npm install -g vercel
```

2. Login to Vercel:

```bash
vercel login
```

3. Deploy:

```bash
vercel
```

4. For production deployment:

```bash
vercel --prod
```

### Using Vercel Dashboard

1. Import your Git repository in the Vercel dashboard
2. Configure the project:
   - Framework preset: Next.js
   - Build command: `npm run build`
   - Output directory: `.next`
   - Environment variables:
     - `NEXT_PUBLIC_API_URL`: URL of your MCTX API (e.g., https://api.mctx-example.com)

3. Deploy the project

## Configuration

### Environment Variables

- `NEXT_PUBLIC_API_URL`: URL of the MCTX API backend
- `NEXT_PUBLIC_DEFAULT_THEME`: Default theme (light or dark)
- `NEXT_PUBLIC_REFRESH_INTERVAL`: Interval for data refresh in milliseconds

### Custom Domain

To set up a custom domain:

1. Go to the Vercel project settings
2. Navigate to the "Domains" tab
3. Add your custom domain
4. Follow the instructions to configure DNS

## Project Structure

```
src/
├── components/          # React components
│   ├── ControlPanel.tsx     # Search parameter controls
│   ├── MetricsVisualization.tsx  # Performance metrics
│   ├── StatsPanel.tsx       # Statistics panel
│   ├── TreeVisualization.tsx  # MCTS tree visualization
│   └── VisualizationTabs.tsx  # Tab navigation
├── lib/                # Utility functions
│   ├── api.ts              # API client
│   └── constants.ts        # Constants and types
├── pages/              # Next.js pages
│   ├── _app.tsx            # App wrapper
│   ├── index.tsx           # Home page
│   └── visualization/[id].tsx  # Single visualization page
└── styles/             # CSS styles
    └── globals.css         # Global styles
```

## Customization

### Styling

The application uses Material UI with a custom theme. You can customize the appearance by modifying the theme in `src/lib/theme.ts`.

### Adding New Visualizations

To add new visualization types:

1. Create a new component in the `components` directory
2. Add it to the `VisualizationTabs.tsx` component
3. Create any necessary API functions in `lib/api.ts`

## Integration with Backend

The frontend communicates with the MCTX API backend through REST endpoints:

- `GET /api/v1/health`: Check API health
- `POST /api/v1/search`: Run a search
- `GET /api/v1/visualization/{id}`: Get visualization data
- `GET /api/v1/metrics`: Get performance metrics

## Browser Compatibility

The application is tested and works with:

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

WebGL support is required for the tree visualization component.

## Troubleshooting

### API Connection Issues

If you're experiencing connection issues:

1. Check that the `NEXT_PUBLIC_API_URL` is correct
2. Verify that CORS is enabled on the API server
3. Check the browser console for error messages

### Visualization Not Rendering

If the visualization doesn't render:

1. Check that WebGL is enabled in your browser
2. Verify that the search ID is valid
3. Check browser console for JavaScript errors

## Performance Optimization

For optimal performance:

1. Use the production build (`npm run build`)
2. Enable Vercel Edge Functions
3. Configure caching headers on API responses
4. Use image optimization for any static assets

## Security Considerations

1. Never store API keys in the frontend code
2. Implement authentication if accessing sensitive data
3. Use HTTPS for all API communication
4. Implement rate limiting on the API

## License

See the project root for license information.

## Contact

For support or questions, contact enterprise@mctx-ai.com