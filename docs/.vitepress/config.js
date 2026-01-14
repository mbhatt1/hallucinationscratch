import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'PCIB Detector',
  description: 'Predictive Coding + Information Bottleneck Hallucination Detection',
  
  themeConfig: {
    logo: '/logo.svg',
    
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'API', link: '/api/detector' },
      { text: 'Research', link: '/research/methodology' },
      { text: 'GitHub', link: 'https://github.com/mbhatt1/hallucinationscratch' }
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Getting Started',
          items: [
            { text: 'Introduction', link: '/guide/introduction' },
            { text: 'Quick Start', link: '/guide/getting-started' },
            { text: 'Installation', link: '/guide/installation' },
            { text: 'Configuration', link: '/guide/configuration' }
          ]
        },
        {
          text: 'Features',
          items: [
            { text: 'Multi-Signal Detection', link: '/guide/multi-signal' },
            { text: 'Trace Validation', link: '/guide/trace-validation' },
            { text: 'Provider Support', link: '/guide/providers' },
            { text: 'MCP Integration', link: '/guide/mcp' }
          ]
        },
        {
          text: 'Advanced',
          items: [
            { text: 'Batch Processing', link: '/guide/batch-processing' },
            { text: 'Custom Signals', link: '/guide/custom-signals' },
            { text: 'Performance Tuning', link: '/guide/performance' }
          ]
        }
      ],
      '/api/': [
        {
          text: 'Core API',
          items: [
            { text: 'PCIBDetector', link: '/api/detector' },
            { text: 'Configuration', link: '/api/config' },
            { text: 'Results', link: '/api/results' }
          ]
        },
        {
          text: 'Backends',
          items: [
            { text: 'OpenAI', link: '/api/backends/openai' },
            { text: 'Anthropic', link: '/api/backends/anthropic' },
            { text: 'Gemini', link: '/api/backends/gemini' }
          ]
        },
        {
          text: 'CLI',
          items: [
            { text: 'Commands', link: '/api/cli' },
            { text: 'MCP Server', link: '/api/mcp-server' }
          ]
        }
      ],
      '/research/': [
        {
          text: 'Theory',
          items: [
            { text: 'Methodology', link: '/research/methodology' },
            { text: 'Predictive Coding', link: '/research/predictive-coding' },
            { text: 'Information Bottleneck', link: '/research/information-bottleneck' }
          ]
        },
        {
          text: 'Evaluation',
          items: [
            { text: 'Benchmarks', link: '/research/benchmarks' },
            { text: 'Ablation Studies', link: '/research/ablation' },
            { text: 'Performance', link: '/research/performance' }
          ]
        },
        {
          text: 'Papers',
          items: [
            { text: 'Main Paper', link: '/research/paper' },
            { text: 'Citations', link: '/research/citations' }
          ]
        }
      ]
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/mbhatt1/hallucinationscratch' }
    ],

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024-present Manish Bhatt'
    },

    search: {
      provider: 'local'
    }
  },

  markdown: {
    lineNumbers: true,
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    }
  }
})
