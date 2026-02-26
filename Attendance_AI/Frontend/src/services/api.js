import axios from 'axios';

const API_BASE_URL = '/api';

const api = axios.create({
    baseURL: API_BASE_URL,
    headers: { 'Content-Type': 'application/json' },
});

// JWT interceptor
api.interceptors.request.use((config) => {
    const token = localStorage.getItem('token');
    if (token) {
        config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
});

api.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response?.status === 401) {
            localStorage.removeItem('token');
            localStorage.removeItem('user');
            window.location.href = '/login';
        }
        return Promise.reject(error);
    }
);

// --- Auth ---
export const authAPI = {
    login: (data) => api.post('/auth/login', data),
    register: (data) => api.post('/auth/register', data),
};

// --- Employees ---
export const employeeAPI = {
    getAll: () => api.get('/employees'),
    getById: (id) => api.get(`/employees/${id}`),
    create: (data) => api.post('/employees', data),
    update: (id, data) => api.put(`/employees/${id}`, data),
    delete: (id) => api.delete(`/employees/${id}`),
    uploadFace: (id, formData) =>
        api.post(`/employees/${id}/upload-face`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        }),
};

// --- Attendance ---
export const attendanceAPI = {
    mark: (data) => api.post('/attendance/mark', data),
    getToday: () => api.get('/attendance/today'),
    getReport: (params) => api.get('/attendance/report', { params }),
    getDashboard: () => api.get('/attendance/dashboard'),
};

// --- Cameras ---
export const cameraAPI = {
    getAll: () => api.get('/cameras'),
    getById: (id) => api.get(`/cameras/${id}`),
    create: (data) => api.post('/cameras', data),
    update: (id, data) => api.put(`/cameras/${id}`, data),
    delete: (id) => api.delete(`/cameras/${id}`),
    // Video upload
    uploadVideo: (formData, onProgress) =>
        api.post('/cameras/upload-video', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
            onUploadProgress: onProgress,
            timeout: 600000, // 10 min timeout for large files
        }),
    getVideos: () => api.get('/cameras/videos'),
    deleteVideo: (fileName) => api.delete(`/cameras/videos/${encodeURIComponent(fileName)}`),
};

export default api;
