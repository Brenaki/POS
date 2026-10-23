// Full-text eligibility probe (PRISMA phase 2), run through the Playwright MCP
// `browser_run_code_unsafe` tool with __ITEMS__ = a slice of fulltext_todo.json
// and __NAME__ = the output name. Opens each record through the CAPES EZproxy
// (DOI resolver, or the IEEE document page), reads the rendered article text
// and records: whether the full text was reachable, term counts, and the
// passages around "oracle" that the eligibility decision is based on.
async (page) => {
  const items = __ITEMS__;
  const proxy = u => u.replace('https://doi.org/', 'https://doi-org.ez82.periodicos.capes.gov.br/')
    .replace('https://ieeexplore.ieee.org/', 'https://ieeexplore-ieee-org.ez82.periodicos.capes.gov.br/');
  const out = {};
  for (const it of items) {
    const url = it.doi ? proxy('https://doi.org/' + it.doi)
      : it.ieee ? proxy('https://ieeexplore.ieee.org/document/' + it.ieee) : null;
    if (!url) { out[it.uid] = { status: 'no_locator' }; continue; }
    try {
      await page.goto(url, { timeout: 45000, waitUntil: 'domcontentloaded' });
      await page.waitForLoadState('networkidle', { timeout: 15000 }).catch(() => {});
      await page.waitForTimeout(2500);
      const text = await page.evaluate(() => document.body.innerText || '');
      const low = text.toLowerCase();
      const count = re => (low.match(re) || []).length;
      const snippets = [];
      let i = low.indexOf('oracle');
      while (i >= 0 && snippets.length < 6) {
        snippets.push(text.slice(Math.max(0, i - 220), i + 220).replace(/\s+/g, ' '));
        i = low.indexOf('oracle', i + 400);
      }
      out[it.uid] = {
        status: 'ok', final_url: page.url(), chars: text.length,
        has_references: /\breferences\b/i.test(text.slice(-Math.min(text.length, 200000))),
        n_oracle: count(/oracle/g), n_upper_bound: count(/upper bound|upper-bound|theoretical limit/g),
        n_diversity: count(/diversity/g), n_double_fault: count(/double[- ]fault/g),
        n_kappa: count(/kappa|q-statistic|disagreement measure/g),
        paywall: /purchase pdf|get access|buy article|access through your institution/i.test(text) && text.length < 25000,
        snippets };
    } catch (e) {
      out[it.uid] = { status: 'error', error: String(e).slice(0, 200) };
    }
  }
  await page.request.post('http://127.0.0.1:8765/save?name=__NAME__', { data: out });
  return Object.entries(out).map(([u, v]) => `${u}:${v.status}:${v.chars || 0}:o${v.n_oracle ?? '-'}`).join(' ');
}
