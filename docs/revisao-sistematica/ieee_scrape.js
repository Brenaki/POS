// IEEE Xplore search-log scraper (PRISMA), run through the Playwright MCP
// `browser_run_code_unsafe` tool. Calls the same REST endpoint the Xplore
// search page uses, from the browser context, over "All Metadata", years
// 2021-2026, 100 rows per page; posts the records to the local receiver
// (raw/__NAME__.json). __QUERY__ is the Scopus string without the
// TITLE-ABS-KEY wrapper and without the year/language filters.
async (page) => {
  const QUERY = '__QUERY__';
  await page.goto('https://ieeexplore.ieee.org/Xplore/home.jsp');
  const rows = [];
  let total = null;
  for (let p = 1; total === null || rows.length < total; p++) {
    const r = await page.request.post('https://ieeexplore.ieee.org/rest/search', {
      headers: { 'Content-Type': 'application/json', Origin: 'https://ieeexplore.ieee.org',
                 Referer: 'https://ieeexplore.ieee.org/search/searchresult.jsp' },
      data: { queryText: QUERY, highlight: false, returnType: 'SEARCH', matchPubs: true,
              rowsPerPage: 100, pageNumber: p, ranges: ['2021_2026_Year'] } });
    const j = await r.json();
    total = j.totalRecords || 0;
    const recs = j.records || [];
    if (!recs.length) break;
    for (const x of recs) rows.push({
      id: x.articleNumber, title: x.articleTitle, doi: x.doi || '',
      authors: (x.authors || []).map(a => a.preferredName).join('; '),
      source: x.publicationTitle, year: x.publicationYear,
      type: x.contentType, abstract: x.abstract || '' });
    if (p > 30) break;
  }
  const data = { source: 'IEEE Xplore', query: QUERY, filters: '2021-2026, All Metadata',
                 total, got: rows.length, retrieved: new Date().toISOString(), rows };
  await page.request.post('http://127.0.0.1:8765/save?name=__NAME__', { data });
  return { total, got: rows.length };
}
