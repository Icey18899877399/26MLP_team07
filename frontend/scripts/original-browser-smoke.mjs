// Full-stack acceptance: actual original images and one complete original rerun.
// Requires a running local stack. No reduced training parameters are submitted.
import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import { readFile } from 'node:fs/promises'

const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL || 'msedge', headless: true })
const origin = process.env.FRONTEND_URL || 'http://127.0.0.1:5173'
const taskNames = { classification: '分类', regression: '回归', clustering: '聚类', anomaly_detection: '异常检测' }
const hash = bytes => createHash('sha256').update(bytes).digest('hex')

try {
  const page = await browser.newPage({ viewport: { width: 1560, height: 1060 }, acceptDownloads: true })
  page.setDefaultTimeout(30000)
  const errors = []
  const legacyCalls = []
  page.on('pageerror', error => errors.push(error.message))
  page.on('request', request => {
    if (/\/api\/(models|datasets|experiments)$/.test(request.url())) legacyCalls.push(request.url())
  })
  await page.goto(origin)
  await page.locator('.task-selector').waitFor()
  const catalogResponse = await page.request.get(`${origin}/api/original-experiments`)
  assert.equal(catalogResponse.status(), 200)
  const catalog = await catalogResponse.json()
  assert.equal(catalog.length, 12)
  assert.equal(catalog.reduce((n, item) => n + item.figures.length, 0), 72)

  for (const item of catalog) {
    await page.locator('.task-selector button').filter({ hasText: taskNames[item.task] }).click()
    await page.locator(`[data-experiment-id="${item.id}"]`).click()
    const archive = page.getByRole('tabpanel', { name: '历史归档原图' })
    assert.equal(await archive.locator('.original-figure-card').count(), 6)
    for (const figure of item.figures) {
      const response = await page.request.get(origin + figure.url)
      assert.equal(response.status(), 200, figure.url)
      const disk = await readFile(new URL(`../../figures/${item.id}/${figure.name}`, import.meta.url))
      assert.equal(hash(await response.body()), hash(disk), `Original image changed: ${item.id}/${figure.name}`)
    }
    for (const file of item.source_data) {
      const response = await page.request.get(origin + file.url)
      assert.equal(response.status(), 200, file.url)
      const disk = await readFile(new URL(`../../figures/${item.id}/${file.name}`, import.meta.url))
      assert.equal(hash(await response.body()), hash(disk))
    }
    console.log(`Original gallery verified: ${item.id} (6 PNG, ${item.source_data.length} source CSV)`)
  }

  await page.locator('.task-selector button').filter({ hasText: taskNames.classification }).click()
  await page.locator('[data-experiment-id="logistic_regression"]').click()
  await page.getByRole('button', { name: /^放大查看/ }).first().click()
  const dialog = page.getByRole('dialog')
  await dialog.waitFor()
  assert.equal(await dialog.locator('img').evaluate(img => img.complete && img.naturalWidth > 0), true)
  const download = page.waitForEvent('download')
  await dialog.getByRole('link', { name: '下载原尺寸 PNG' }).click()
  assert.match((await download).suggestedFilename(), /\.png$/)
  await page.keyboard.press('Escape')

  const submitted = page.waitForResponse(response => response.url().endsWith('/api/original-runs') && response.request().method() === 'POST')
  await page.locator('.original-reproduction-details > summary').click()
  await page.getByRole('button', { name: '启动原实验', exact: true }).click()
  const response = await submitted
  assert.equal(response.status(), 202)
  assert.deepEqual(response.request().postDataJSON(), { experiment_id: 'logistic_regression' })
  let run = await response.json()
  const deadline = Date.now() + 300000
  while (['queued', 'running'].includes(run.status) && Date.now() < deadline) {
    await new Promise(resolve => setTimeout(resolve, 1500))
    run = await (await page.request.get(`${origin}/api/original-runs/${run.run_id}`)).json()
  }
  assert.equal(run.status, 'completed', JSON.stringify(run))
  assert.equal(run.figures.length, 6)
  assert.equal(run.source_data.length, 6)
  for (const figure of run.figures) {
    assert.ok(figure.url.includes(`/original-runs/${run.run_id}/assets/`))
    const result = await page.request.get(origin + figure.url)
    assert.equal(result.status(), 200)
    assert.ok((await result.body()).subarray(0, 4).equals(Buffer.from([137, 80, 78, 71])))
  }
  await page.reload()
  await page.locator('[data-experiment-id="logistic_regression"]').click()
  await page.getByRole('tab', { name: '本次复现产物', exact: true }).click()
  const newResults = page.getByRole('tabpanel', { name: '本次复现产物' })
  await newResults.locator('.original-figure-card').first().waitFor()
  assert.equal(await newResults.locator('.original-figure-card').count(), 6)
  assert.match(await newResults.innerText(), new RegExp(run.run_id))

  await page.goto(`${origin}/datasets`)
  await page.getByRole('heading', { name: '原始数据目录', exact: true }).waitFor()
  const datasets = await (await page.request.get(`${origin}/api/original-datasets`)).json()
  assert.equal(datasets.length, 7)
  assert.equal(datasets.filter(item => item.archived_only).length, 2)
  await page.goto(`${origin}/benchmark`)
  await page.getByText(run.run_id, { exact: true }).first().waitFor()

  await page.setViewportSize({ width: 390, height: 844 })
  await page.getByRole('button', { name: '打开导航', exact: true }).click()
  await page.getByRole('navigation', { name: '主导航' }).getByRole('link', { name: '原实验图谱', exact: true }).click()
  await page.locator('.task-selector').waitFor()
  assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true, 'Mobile viewport overflows')
  assert.deepEqual(errors, [])
  assert.deepEqual(legacyCalls, [], 'Primary UI used legacy peer registry/experiment API')
  console.log(`PASS: all 72 original PNG and 18 CSV bytes, 12 galleries, original PNG download, full logistic rerun ${run.run_id}, persistence, 7 datasets, mobile navigation; no legacy API calls`)
} finally {
  await browser.close()
}
