export default async function handler(req, res) {
  const job_id = typeof req.query.job_id === 'string' ? req.query.job_id : ''
  if (!job_id) {
    res.status(400).send('job_id required')
    return
  }
  const base = process.env.BACKEND_URL
  if (!base) {
    res.status(500).json({ error: 'BACKEND_URL not set' })
    return
  }
  const r = await fetch(`${base}/discover/jobs/${job_id}/stream`)
  res.setHeader('Content-Type', 'text/event-stream')
  res.setHeader('Cache-Control', 'no-cache')
  res.setHeader('Connection', 'keep-alive')
  const reader = r.body.getReader()
  for (; ;) {
    const { value, done } = await reader.read()
    if (done) break
    res.write(value)
  }
  res.end()
}