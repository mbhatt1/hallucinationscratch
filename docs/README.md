# PCIB Detector Documentation

This directory contains the VitePress documentation for PCIB Detector.

## Development

### Prerequisites

- Node.js 16+ 
- npm or yarn

### Installation

```bash
# Install dependencies
npm install

# Or with yarn
yarn install
```

### Running Locally

```bash
# Start dev server (hot reload)
npm run docs:dev

# Or with yarn
yarn docs:dev
```

The docs will be available at `http://localhost:5173`

### Building for Production

```bash
# Build static site
npm run docs:build

# Preview production build
npm run docs:preview
```

The built site will be in `docs/.vitepress/dist/`

## Structure

```
docs/
├── .vitepress/          # VitePress configuration
│   ├── config.js       # Main config
│   └── theme/          # Custom theme
│       ├── index.js    # Theme entry
│       └── custom.css  # Custom styles
├── guide/              # User guides
│   ├── introduction.md
│   ├── getting-started.md
│   ├── multi-signal.md
│   └── ...
├── api/                # API reference
│   ├── detector.md
│   ├── config.md
│   └── ...
├── research/           # Research papers
│   ├── methodology.md
│   ├── benchmarks.md
│   └── ...
└── index.md           # Homepage
```

## Adding Content

### New Guide

1. Create `docs/guide/new-guide.md`
2. Add to sidebar in `docs/.vitepress/config.js`:

```js
{
  text: 'Guides',
  items: [
    // ...
    { text: 'New Guide', link: '/guide/new-guide' }
  ]
}
```

### New API Doc

1. Create `docs/api/new-api.md`
2. Add to API sidebar in config

### Markdown Features

VitePress supports:
- GitHub-flavored Markdown
- Syntax highlighting
- Custom containers (tip, warning, danger)
- Math equations (KaTeX)
- Code groups
- Mermaid diagrams

## Deployment

### GitHub Pages

```yaml
# .github/workflows/deploy.yml
name: Deploy Docs
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: 18
      - run: npm install
      - run: npm run docs:build
      - uses: peaceiris/actions-gh-pages@v3
        with:
          github_token: ${{ secrets.GITHUB_TOKEN }}
          publish_dir: docs/.vitepress/dist
```

### Netlify

1. Connect repository
2. Build command: `npm run docs:build`
3. Publish directory: `docs/.vitepress/dist`

### Vercel

1. Import project
2. Framework: VitePress
3. Build command: `npm run docs:build`
4. Output directory: `docs/.vitepress/dist`

## Contributing

See [CONTRIBUTING.md](../pcib_detector/CONTRIBUTING.md) for contribution guidelines.
