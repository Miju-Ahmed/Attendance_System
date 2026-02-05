import { useState, useEffect } from 'react';
import { DashboardLayout } from '../layout/DashboardLayout';
import { userService, adminService } from '../../services/api';

export const AdminDashboard = () => {
    const [stats, setStats] = useState({ users: 0, assets: 0, leaves: 0 });
    const [loading, setLoading] = useState(true);
    const [users, setUsers] = useState([]);
    const [logs, setLogs] = useState([]);
    const [syncing, setSyncing] = useState(false);
    const [uploading, setUploading] = useState(false);
    const [processedVideo, setProcessedVideo] = useState(null);
    const [message, setMessage] = useState({ type: '', text: '' });

    useEffect(() => {
        fetchData();
        // Poll logs every 5 seconds for live updates
        const interval = setInterval(fetchLogs, 5000);
        return () => clearInterval(interval);
    }, []);

    const fetchData = async () => {
        try {
            const usersData = await adminService.getUsersWithSecrets();
            setUsers(usersData);
            setStats({ users: usersData.length, assets: 0, leaves: 0 });
            await fetchLogs();
        } catch (error) {
            console.error('Failed to fetch data');
        } finally {
            setLoading(false);
        }
    };

    const fetchLogs = async () => {
        try {
            const logsData = await adminService.getAttendanceLogs();
            setLogs(logsData);
        } catch (error) {
            console.error('Failed to fetch logs');
        }
    };

    const handleSync = async () => {
        try {
            setSyncing(true);
            const result = await adminService.syncUsers();
            setMessage({ type: 'success', text: `Sync complete! Added: ${result.added}` });
            fetchData();
        } catch (error) {
            setMessage({ type: 'error', text: 'Sync failed' });
        } finally {
            setSyncing(false);
        }
    };

    const handleFileUpload = async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        try {
            setUploading(true);
            setProcessedVideo(null); // Reset previous video
            const response = await adminService.uploadVideo(file);
            setMessage({ type: 'success', text: response.message || 'Video uploaded successfully' });
            if (response.processedVideo) {
                setProcessedVideo(response.processedVideo);
            }
        } catch (error) {
            setMessage({ type: 'error', text: 'Upload failed' });
        } finally {
            setUploading(false);
        }
    };

    const handleLiveStream = async () => {
        try {
            const response = await adminService.startLiveStream();
            setMessage({ type: 'success', text: response });
        } catch (error) {
            setMessage({ type: 'error', text: 'Failed to start live stream' });
        }
    };

    return (
        <DashboardLayout>
            <div className="space-y-6">
                <div className="flex justify-between items-center">
                    <div>
                        <h1 className="text-3xl font-bold text-gray-900">Admin Dashboard</h1>
                        <p className="text-gray-600 mt-2">System overview and management</p>
                    </div>
                    {message.text && (
                        <div className={`px-4 py-2 rounded ${message.type === 'success' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                            {message.text}
                        </div>
                    )}
                </div>

                {/* Stats Grid */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div className="card bg-blue-600 text-white">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-blue-100">Total Users</p>
                                <p className="text-3xl font-bold">{stats.users}</p>
                            </div>
                            <div className="text-4xl">👥</div>
                        </div>
                    </div>
                    <div className="card bg-indigo-600 text-white">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-indigo-100">Daily Events</p>
                                <p className="text-3xl font-bold">{logs.length}</p>
                            </div>
                            <div className="text-4xl">📋</div>
                        </div>
                    </div>
                    <div className="card bg-green-600 text-white">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-green-100">AI Status</p>
                                <p className="text-3xl font-bold">Active</p>
                            </div>
                            <div className="text-4xl">🤖</div>
                        </div>
                    </div>
                </div>

                {/* AI & Data Management Section */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="card">
                        <h3 className="text-lg font-semibold mb-4">AI Data Management</h3>
                        <div className="space-y-4">
                            <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                                <div>
                                    <p className="font-medium">Sync Employees</p>
                                    <p className="text-sm text-gray-500">Import users from Face Embeddings DB</p>
                                </div>
                                <button
                                    onClick={handleSync}
                                    disabled={syncing}
                                    className="btn btn-primary"
                                >
                                    {syncing ? 'Syncing...' : 'Sync Now'}
                                </button>
                            </div>
                        </div>
                    </div>

                    <div className="card">
                        <h3 className="text-lg font-semibold mb-4">Video Attendance</h3>
                        <div className="space-y-4">
                            <div className="p-4 bg-gray-50 rounded-lg">
                                <p className="font-medium mb-2">Upload Video for Processing</p>
                                <div className="space-y-4">
                                    <div className="flex items-center space-x-4">
                                        <input
                                            type="file"
                                            accept="video/*"
                                            onChange={handleFileUpload}
                                            disabled={uploading}
                                            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                                        />
                                        {uploading && <p className="text-sm text-blue-600">Processing...</p>}
                                    </div>
                                    {processedVideo && (
                                        <div className="mt-4">
                                            <p className="font-medium mb-2 text-green-700">Processed Video Output:</p>
                                            <video
                                                controls
                                                className="w-full rounded-lg shadow-lg border border-gray-200"
                                                src={`http://localhost:8081/uploads/${processedVideo}`}
                                            >
                                                Your browser does not support the video tag.
                                            </video>
                                            <p className="text-xs text-gray-500 mt-1">If video doesn't play immediately, it might still be processing. Wait a moment and refresh/play.</p>
                                        </div>
                                    )}
                                </div>
                            </div>
                            <div className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                                <div>
                                    <p className="font-medium">Live Camera</p>
                                    <p className="text-sm text-gray-500">Start server-side live stream window</p>
                                </div>
                                <button
                                    onClick={handleLiveStream}
                                    className="btn btn-secondary bg-red-600 text-white hover:bg-red-700"
                                >
                                    Start Live Camera
                                </button>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Daily Attendance Logs */}
                <div className="card">
                    <h3 className="text-lg font-semibold mb-4">Daily Attendance Logs (Last 100 Events)</h3>
                    <div className="overflow-x-auto max-h-96">
                        <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50 sticky top-0">
                                <tr>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Time</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Event</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Confidence</th>
                                </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                                {logs.length > 0 ? (
                                    logs.map((log) => (
                                        <tr key={log.id}>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                {new Date(log.timestamp).toLocaleString()}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <div className="text-sm font-medium text-gray-900">{log.person_name}</div>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <span className={`px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full 
                                                    ${log.event_type === 'ENTRY' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
                                                    {log.event_type}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                {(log.confidence * 100).toFixed(1)}%
                                            </td>
                                        </tr>
                                    ))
                                ) : (
                                    <tr>
                                        <td colSpan="4" className="px-6 py-4 text-center text-sm text-gray-500">
                                            No attendance records found yet.
                                        </td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* User List with Secrets */}
                <div className="card">
                    <h3 className="text-lg font-semibold mb-4">User Database (Sensitive Info)</h3>
                    <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50">
                                <tr>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Name</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Email</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Password</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Stable ID</th>
                                </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                                {users.map((user) => (
                                    <tr key={user.id}>
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            <div className="text-sm font-medium text-gray-900">{user.name}</div>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            <div className="text-sm text-gray-500">{user.email}</div>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            <span className="px-2 py-1 inline-flex text-xs leading-5 font-semibold rounded-full bg-red-100 text-red-800">
                                                {user.visiblePassword || 'Hash only'}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                            {user.stableId || '-'}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </DashboardLayout>
    );
};
