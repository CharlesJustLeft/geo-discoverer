export default async function handler(req, res) {
  console.log('Function started');
  if (req.method !== 'POST') {
    res.status(405).send('Method Not Allowed')
    return
  }
  const base = process.env.BACKEND_URL
  console.log('BACKEND_URL:', base);

  if (!base) {
    console.error('BACKEND_URL missing');
    res.status(500).json({ error: 'BACKEND_URL not set' })
    return
  }

  try {
    const target = `${base}/discover/jobs`;
    console.log('Fetching:', target);

    const r = await fetch(target, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req.body || {}),
    })

    console.log('Response status:', r.status);
    const data = await r.json()
    res.status(r.status).json(data)
  } catch (err) {
    console.error('Fetch error:', err);
    res.status(500).json({ error: String(err) })
  }
}