export default async function handler(req, res) {
  const { job_id } = req.query;
  
  if (!job_id) {
    return res.status(400).json({ error: 'job_id required' });
  }

  const backendUrl = process.env.BACKEND_URL || 'https://geo-discoverer-production.up.railway.app';
  
  try {
    const response = await fetch(`${backendUrl}/discover/jobs/${job_id}/partial`);
    const data = await response.json();
    return res.status(response.status).json(data);
  } catch (error) {
    console.error('Partial result fetch error:', error);
    return res.status(500).json({ error: 'Failed to fetch partial results' });
  }
}

