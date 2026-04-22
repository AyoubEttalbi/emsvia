import axios from 'axios';

const api = axios.create({
    baseURL: 'http://127.0.0.1:8000/api',
});

export const getStats = () => api.get('/stats');
export const getStudents = () => api.get('/students/');
export const getAttendance = (params) => api.get('/attendance/records', { params });
export const getRecentAttendance = () => api.get('/attendance/recent');
export const getPendingUnknowns = () => api.get('/unknown/pending');
export const getHealth = () => api.get('/health');
export const getLogs = () => api.get('/system/logs');
export const rebuildEmbeddings = () => api.post('/embeddings/rebuild');

export const reviewUnknown = (id) => api.patch(`/unknown/${id}/review`);


export const dismissUnknown = (id) => api.delete(`/unknown/${id}`);

export default api;

