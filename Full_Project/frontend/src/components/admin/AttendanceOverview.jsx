import { useState, useEffect } from 'react';
import { DashboardLayout } from '../layout/DashboardLayout';
import { attendanceService } from '../../services/api';

export const AttendanceOverview = () => {
    const [attendance, setAttendance] = useState([]);
    const [loading, setLoading] = useState(true);
    const [message, setMessage] = useState('');

    useEffect(() => {
        fetchAttendance();
    }, []);

    const fetchAttendance = async () => {
        try {
            const data = await attendanceService.getAllAttendance();
            setAttendance(data);
        } catch (error) {
            console.error('Failed to fetch attendance');
        } finally {
            setLoading(false);
        }
    };

    const handleSync = async () => {
        setLoading(true);
        setMessage('');
        try {
            const result = await attendanceService.syncFromSQLite();
            setMessage(result.message);
            fetchAttendance();
        } catch (error) {
            setMessage('Failed to sync attendance data');
        } finally {
            setLoading(false);
        }
    };

    return (
        <DashboardLayout>
            <div>
                <div className="flex justify-between items-center mb-6">
                    <h1 className="text-3xl font-bold text-gray-900">Attendance Overview</h1>
                    <button onClick={handleSync} disabled={loading} className="btn-primary disabled:opacity-50">
                        {loading ? 'Syncing...' : 'Sync from AI System'}
                    </button>
                </div>

                {message && (
                    <div className={`mb-4 p-4 rounded-lg ${message.includes('success') || message.includes('Synced') ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                        {message}
                    </div>
                )}

                <div className="card">
                    <h2 className="text-xl font-semibold text-gray-900 mb-4">Recent Attendance Records</h2>
                    {attendance.length === 0 ? (
                        <p className="text-gray-500">No attendance records yet. Click "Sync from AI System" to import data.</p>
                    ) : (
                        <div className="overflow-x-auto">
                            <table className="min-w-full divide-y divide-gray-200">
                                <thead className="bg-gray-50">
                                    <tr>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Employee</th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Event</th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Timestamp</th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Confidence</th>
                                    </tr>
                                </thead>
                                <tbody className="bg-white divide-y divide-gray-200">
                                    {attendance.slice(0, 50).map((record) => (
                                        <tr key={record.id}>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                                                {record.user?.name || `User #${record.stableId}`}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <span className={`px-2 py-1 text-xs font-semibold rounded-full ${record.eventType === 'ENTRY' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                                                    }`}>
                                                    {record.eventType}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                {new Date(record.timestamp).toLocaleString()}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                {record.confidence ? `${(record.confidence * 100).toFixed(1)}%` : 'N/A'}
                                            </td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            </div>
        </DashboardLayout>
    );
};
