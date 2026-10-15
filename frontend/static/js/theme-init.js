// Applied before first paint so a saved manual theme choice doesn't flash the
// system-default theme for a frame; ui.js's initChrome() is the source of truth
// after that. A file (not an inline script) so the page's CSP can forbid inline script.
try {
  var t = localStorage.getItem('pai_theme');
  if (t === 'dark' || t === 'light') document.documentElement.setAttribute('data-theme', t);
} catch (e) {}
