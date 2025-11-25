export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }
  
  const { job_id } = req.query;
  if (!job_id) {
    return res.status(400).json({ error: 'job_id required' });
  }

  const backendUrl = process.env.BACKEND_URL || 'https://geo-discoverer-production.up.railway.app';
  
  try {
    const response = await fetch(`${backendUrl}/discover/jobs/${job_id}/cancel`, {
      method: 'POST'
    });
    const data = await response.json();
    return res.status(response.status).json(data);
  } catch (error) {
    console.error('Cancel error:', error);
    return res.status(500).json({ error: 'Failed to cancel job' });
  }
}

