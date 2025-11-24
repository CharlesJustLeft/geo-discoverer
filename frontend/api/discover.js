export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.status(405).send('Method Not Allowed')
    return
  }
  const base = process.env.BACKEND_URL
  if (!base) {
    res.status(500).json({ error: 'BACKEND_URL not set' })
    return
  }
  const r = await fetch(`${base}/discover/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req.body || {}),
  })
  const data = await r.json()
  res.status(r.status).json(data)
}