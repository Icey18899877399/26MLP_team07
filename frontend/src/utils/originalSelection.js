export function createRunLinkResolver() {
  let pendingId = ''
  return {
    setPending(id) { pendingId = typeof id === 'string' ? id : '' },
    cancel() { pendingId = '' },
    take(runs) {
      if (!pendingId) return null
      const match = runs.find(run => run.run_id === pendingId) || null
      if (match) pendingId = ''
      return match
    }
  }
}
