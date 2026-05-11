import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// ─────────────────────────────────────────────
// Query Endpoints
// ─────────────────────────────────────────────

export const queryStandardRAG = async (question, topK = 5) => {
  const response = await api.post('/rag', {
    question,
    top_k: topK,
  });
  return response.data;
};

export const queryWorkflowAgent = async (question, sessionId = null, topK = 5) => {
  const response = await api.post('/workflow-agent', {
    question,
    session_id: sessionId,
    top_k: topK,
  });
  return response.data;
};

// ─────────────────────────────────────────────
// Session Management
// ─────────────────────────────────────────────

export const getSessionState = async (sessionId) => {
  const response = await api.get(`/session/${sessionId}/state`);
  return response.data;
};

export const resetSession = async (sessionId) => {
  const response = await api.post(`/session/${sessionId}/reset`);
  return response.data;
};

// ─────────────────────────────────────────────
// Data Management
// ─────────────────────────────────────────────

export const getDataStatus = async () => {
  const response = await api.get('/data-status');
  return response.data;
};

export const populateData = async () => {
  const response = await api.post('/populate');
  return response.data;
};

export const getPopulateStatus = async () => {
  const response = await api.get('/populate/status');
  return response.data;
};

// ─────────────────────────────────────────────
// Metrics
// ─────────────────────────────────────────────

export const getMetrics = async () => {
  const response = await api.get('/metrics');
  return response.data;
};

export const resetMetrics = async () => {
  const response = await api.post('/metrics/reset');
  return response.data;
};

// ─────────────────────────────────────────────
// Health Check
// ─────────────────────────────────────────────

export const healthCheck = async () => {
  const response = await api.get('/health');
  return response.data;
};

export default api;

// Made with Bob
