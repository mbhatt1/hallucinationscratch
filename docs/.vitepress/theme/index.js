// Custom theme extending default VitePress theme
import DefaultTheme from 'vitepress/theme'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app, router, siteData }) {
    // Custom app-level enhancements can go here
  }
}
