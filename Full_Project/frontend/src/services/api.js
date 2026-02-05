import axiosInstance from '../utils/axios';

export const userService = {
    getAllUsers: async () => {
        const response = await axiosInstance.get('/users');
        return response.data;
    },

    getUserById: async (id) => {
        const response = await axiosInstance.get(`/users/${id}`);
        return response.data;
    },

    updateUser: async (id, userData) => {
        const response = await axiosInstance.put(`/users/${id}`, userData);
        return response.data;
    },

    uploadProfilePhoto: async (id, file) => {
        const formData = new FormData();
        formData.append('file', file);
        const response = await axiosInstance.post(`/users/${id}/photo`, formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        return response.data;
    },

    getProfilePhotoUrl: (filename) => {
        return `http://localhost:8081/api/users/photo/${filename}`;
    },
};

export const adminService = {
    getUsersWithSecrets: async () => {
        const response = await axiosInstance.get('/admin/users-with-secrets');
        return response.data;
    },

    syncUsers: async () => {
        const response = await axiosInstance.post('/admin/sync-users');
        return response.data;
    },

    uploadVideo: async (file) => {
        const formData = new FormData();
        formData.append('file', file);
        const response = await axiosInstance.post('/admin/upload-video', formData, {
            headers: { 'Content-Type': 'multipart/form-data' },
        });
        return response.data;
    },

    getAttendanceLogs: async () => {
        const response = await axiosInstance.get('/admin/attendance-logs');
        return response.data;
    },

    startLiveStream: async () => {
        const response = await axiosInstance.post('/admin/live-stream');
        return response.data;
    },
};

export const assetService = {
    getAllAssets: async () => {
        const response = await axiosInstance.get('/assets');
        return response.data;
    },

    createAsset: async (assetData) => {
        const response = await axiosInstance.post('/assets', assetData);
        return response.data;
    },

    requestAsset: async (id) => {
        const response = await axiosInstance.post(`/assets/${id}/request`);
        return response.data;
    },

    getMyAssets: async () => {
        const response = await axiosInstance.get('/assets/my-assets');
        return response.data;
    },
};

export const leaveService = {
    getAllLeaves: async () => {
        const response = await axiosInstance.get('/leaves');
        return response.data;
    },

    getPendingLeaves: async () => {
        const response = await axiosInstance.get('/leaves/pending');
        return response.data;
    },

    applyLeave: async (leaveData) => {
        const response = await axiosInstance.post('/leaves', leaveData);
        return response.data;
    },

    approveLeave: async (id) => {
        const response = await axiosInstance.put(`/leaves/${id}/approve`);
        return response.data;
    },

    rejectLeave: async (id) => {
        const response = await axiosInstance.put(`/leaves/${id}/reject`);
        return response.data;
    },

    getMyLeaves: async () => {
        const response = await axiosInstance.get('/leaves/my-leaves');
        return response.data;
    },
};

export const attendanceService = {
    getAllAttendance: async () => {
        const response = await axiosInstance.get('/attendance');
        return response.data;
    },

    getMyAttendance: async () => {
        const response = await axiosInstance.get('/attendance/my-attendance');
        return response.data;
    },

    syncFromSQLite: async () => {
        const response = await axiosInstance.post('/attendance/sync');
        return response.data;
    },

    getAttendanceByUserId: async (userId) => {
        const response = await axiosInstance.get(`/attendance/user/${userId}`);
        return response.data;
    },
};
