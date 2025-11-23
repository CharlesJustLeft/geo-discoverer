export default async function handler(req, res) {
  const job_id = typeof req.query.job_id === 'string' ? req.query.job_id : ''
  if (!job_id) {
    res.status(400).send('job_id required')
    return
  }
  const base = process.env.SPOONOS_BASE_URL
  if (!base) {
    res.status(500).json({ error: 'SPOONOS_BASE_URL not set' })
    return
  }
  const r = await fetch(`${base}/discover/jobs/${job_id}/result`)
  const data = await r.json()
  res.status(r.status).json(data)
}