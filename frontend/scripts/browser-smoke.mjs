// Install Playwright locally or set PLAYWRIGHT_MODULE to its file:// module URL.
import assert from 'node:assert/strict'
const {chromium}=await import(process.env.PLAYWRIGHT_MODULE || 'playwright')
const browser=await chromium.launch({channel:process.env.BROWSER_CHANNEL || 'msedge',headless:true})
try {
  const page=await browser.newPage({viewport:{width:1440,height:1080},acceptDownloads:true})
  page.setDefaultTimeout(30000)
  const errors=[]
  page.on('pageerror',error=>errors.push(error.message))
  const origin=process.env.FRONTEND_URL || 'http://127.0.0.1:5173'
  await page.goto(origin)
  await page.getByText('已连接',{exact:true}).waitFor()
  await page.getByRole('button',{name:'开始训练',exact:true}).waitFor()
  const catalog=await (await page.request.get(`${origin}/api/models`)).json()
  assert.equal(catalog.length,12)
  for (const model of catalog) {
    await page.locator(`[data-model-id="${model.id}"]`).click()
    const response=page.waitForResponse(r=>r.url().endsWith('/api/experiments') && r.request().method()==='POST',{timeout:300000})
    await page.getByRole('button',{name:'开始训练',exact:true}).click()
    const result=await response
    assert.equal(result.status(),200,await result.text())
    const data=await result.json()
    await page.locator('.run-identity').filter({hasText:data.run_id}).waitFor()
    assert.equal(await page.locator('.chart-card').count(),data.metadata.visualizations.length)
    for (const chart of data.metadata.visualizations) {
      await page.locator(`[data-chart-id="${chart.id}"] canvas`).first().waitFor({state:'attached'})
    }
    if (model.id==='cart_decision_tree.optimized') {
      await page.getByRole('button',{name:'点击放大查看',exact:true}).first().click()
      await page.getByRole('dialog').waitFor()
      const download=page.waitForEvent('download')
      await page.getByRole('button',{name:'下载高清 PNG',exact:true}).click()
      assert.match((await download).suggestedFilename(),/\.png$/)
      await page.getByRole('dialog').getByRole('button',{name:'关闭',exact:true}).click()
    }
    console.log(`Browser passed: ${model.id}, ${data.metadata.visualizations.length} charts`)
  }
  const jsonDownload=page.waitForEvent('download')
  await page.getByRole('button',{name:'导出完整结果',exact:true}).click()
  assert.match((await jsonDownload).suggestedFilename(),/\.json$/)
  await page.getByRole('button',{name:'载入参数',exact:true}).first().click()
  await page.reload()
  await page.locator('.history-table .el-table__row').first().waitFor()
  await page.getByRole('menuitem',{name:'数据集',exact:true}).click()
  await page.getByRole('heading',{name:'可用数据集',exact:true}).waitFor()
  assert.equal(await page.locator('.dataset-grid article').count(),6)
  await page.getByRole('menuitem',{name:'算法对比',exact:true}).click()
  await page.getByRole('heading',{name:'在相同条件下，比较模型',exact:true}).waitFor()
  await page.locator('.base-chart canvas').first().waitFor({state:'attached'})
  await page.getByRole('menuitem',{name:'使用手册',exact:true}).click()
  await page.getByRole('heading',{name:'从第一次训练开始',exact:true}).waitFor()
  await page.setViewportSize({width:390,height:844})
  await page.getByRole('menuitem',{name:'实验工作台',exact:true}).click()
  await page.getByRole('heading',{name:'让每一次实验，都看得见',exact:true}).waitFor()
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth<=window.innerWidth),true,'Mobile page overflows horizontally')
  assert.deepEqual(errors,[])
  console.log('PASS: 12 real browser runs, chart enlargement/download, JSON export, persisted history, navigation and mobile layout')
} finally {await browser.close()}
