// Scopus (via CAPES/EZproxy) result-list scraper for the PRISMA search log.
//
// Run through the Playwright MCP `browser_run_code_unsafe` tool with __QUERY__
// replaced by the exact Scopus advanced query (queries/*.txt). Export needs a
// personal Elsevier account and the `offset` URL parameter is overridden by
// the session, so the scraper sets "Display: 200 results", opens every
// abstract and reads the result table. The MCP sandbox has no `require`, so
// the JSON is POSTed to a local receiver that writes raw/__NAME__.json; only
// the counts come back to the caller.
async (page) => {
  const QUERY = '__QUERY__';
  const url = 'https://www-scopus-com.ez82.periodicos.capes.gov.br/results/results.uri'
    + '?src=s&sot=a&sdt=a&sort=plf-f&s=' + encodeURIComponent(QUERY);
  const count = () => page.evaluate(() =>
    document.querySelectorAll('tbody tr a[href*="/pages/publications/"]').length);
  await page.goto(url);
  await page.getByText(/documents? found/).first().waitFor({ timeout: 60000 });
  const txt = await page.evaluate(() => document.body.innerText);
  const total = parseInt((txt.match(/([\d,]+) documents? found/) || [, '0'])[1].replace(/,/g, ''));
  if (total > 0) {
    await page.waitForFunction(() => document.querySelectorAll(
      'tbody tr a[href*="/pages/publications/"]').length > 0, null, { timeout: 60000 });
    const display = page.locator('select').filter({ hasText: '200 results' });
    await display.selectOption('200');
    const want = Math.min(200, total);
    await page.waitForFunction(n => document.querySelectorAll(
      'tbody tr a[href*="/pages/publications/"]').length >= n, want, { timeout: 60000 })
      .catch(() => {});
    const toggle = page.getByRole('button', { name: 'Show all abstracts' });
    if (await toggle.count()) { await toggle.first().click(); await page.waitForTimeout(3000); }
  }
  const rows = await page.evaluate(() => {
    const out = [];
    let cur = null;
    for (const tr of document.querySelectorAll('tbody tr')) {
      const a = tr.querySelector('a[href*="/pages/publications/"]');
      const td = [...tr.querySelectorAll('td')].map(t => t.innerText.replace(/\s+/g, ' ').trim());
      if (a) {
        cur = { eid: (a.href.match(/publications\/(\d+)/) || [])[1], title: a.innerText.trim(),
                authors: td[2], source: td[3], year: td[4], cited: td[5], abstract: '' };
        out.push(cur);
      } else if (cur) {
        const t = tr.innerText.replace(/\s+/g, ' ').trim();
        if (t.length > cur.abstract.length) cur.abstract = t;
      }
    }
    return out;
  });
  const data = { query: QUERY, total, got: rows.length, shown: await count(),
                 retrieved: new Date().toISOString(), rows };
  await page.request.post('http://127.0.0.1:8765/save?name=__NAME__', { data });
  return { total, got: rows.length, shown: data.shown,
           with_abstract: rows.filter(r => r.abstract.length > 200).length };
}
