const BACKEND_URL = process.env.BACKEND_URL || 'https://geo-discoverer-production.up.railway.app';

export default async function handler(req, res) {
  const { job_id } = req.query;
  
  if (!job_id) {
    return res.status(400).json({ error: 'job_id required' });
  }
  
  try {
    const response = await fetch(`${BACKEND_URL}/discover/jobs/${job_id}/queue`);
    const data = await response.json();
    
    if (!response.ok) {
      return res.status(response.status).json(data);
    }
    
    return res.status(200).json(data);
  } catch (error) {
    return res.status(500).json({ error: 'Failed to fetch queue position' });
  }
}

