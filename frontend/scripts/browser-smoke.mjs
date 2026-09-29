// Optional: install playwright locally, or provide PLAYWRIGHT_MODULE as an import URL.
import assert from 'node:assert/strict'
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL || 'msedge', headless: true })
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1080 } })
  const errors = []
  page.on('pageerror', error => { errors.push(error.message); console.error('Page error:', error.message) })
  await page.goto('http://127.0.0.1:5173')
  await page.getByText('已连接', { exact: true }).waitFor()
  for (const name of ['Optimized K-Means', 'Optimized Logistic Regression']) {
    await page.locator('.algo-item').filter({ hasText: name }).click()
    const response = page.waitForResponse(r => r.url().endsWith('/api/experiments') && r.request().method() === 'POST')
    await page.getByRole('button', { name: '开始训练', exact: true }).click()
    if (name === 'Optimized Logistic Regression') {
      await page.getByRole('button', { name: '清空', exact: true }).click()
      assert.equal(await page.locator('.run-monitor').count(), 1, 'Clearing results must preserve the active run')
    }
    const result = await response
    assert.equal(result.status(), 200, await result.text())
    await page.getByText('已完成', { exact: true }).waitFor()
    console.log('Browser experiment passed:', name)
  }
  await page.getByRole('menuitem', { name: '数据集', exact: true }).click()
  await page.getByRole('heading', { name: '可用数据集', exact: true }).waitFor()
  await page.getByText('569', { exact: true }).waitFor()
  await page.getByRole('menuitem', { name: '算法对比', exact: true }).click()
  await page.getByText('选择已运行的数据集', { exact: true }).click()
  await page.getByRole('option', {name: 'wdbc', exact: true}).click()
  await page.getByText('Optimized Logistic Regression', { exact: true }).waitFor()
  assert.deepEqual(errors, [])
  console.log('Browser navigation and actual results passed; no uncaught page errors')
} finally { await browser.close() }
