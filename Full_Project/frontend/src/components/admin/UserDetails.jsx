import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { DashboardLayout } from '../layout/DashboardLayout';
import { userService, attendanceService } from '../../services/api';

export const UserDetails = () => {
    const { userId } = useParams();
    const [user, setUser] = useState(null);
    const [attendance, setAttendance] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [userData, attendanceData] = await Promise.all([
                    userService.getUserById(userId),
                    attendanceService.getAttendanceByUserId(userId)
                ]);
                setUser(userData);
                setAttendance(attendanceData);
            } catch (error) {
                console.error('Failed to fetch user details:', error);
            } finally {
                setLoading(false);
            }
        };

        if (userId) {
            fetchData();
        }
    }, [userId]);

    if (loading) {
        return (
            <DashboardLayout>
                <div className="flex justify-center items-center h-64">
                    <div className="text-gray-500 text-lg">Loading details...</div>
                </div>
            </DashboardLayout>
        );
    }

    if (!user) {
        return (
            <DashboardLayout>
                <div className="flex flex-col items-center justify-center h-64">
                    <h2 className="text-2xl font-bold text-gray-800 mb-2">User Not Found</h2>
                    <p className="text-gray-600 mb-4">The requested user could not be found.</p>
                    <Link to="/admin/users" className="text-primary-600 hover:text-primary-800">
                        Back to User List
                    </Link>
                </div>
            </DashboardLayout>
        );
    }

    return (
        <DashboardLayout>
            <div className="mb-6">
                <Link to="/admin/users" className="text-gray-500 hover:text-gray-700 flex items-center mb-4">
                    <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
                    </svg>
                    Back to Users
                </Link>

                <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden mb-8">
                    <div className="px-6 py-5 border-b border-gray-200 bg-gray-50">
                        <h3 className="text-lg font-medium text-gray-900">User Profile</h3>
                    </div>
                    <div className="px-6 py-5">
                        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <div>
                                <label className="block text-sm font-medium text-gray-500 mb-1">Full Name</label>
                                <div className="text-gray-900 font-medium text-lg">{user.name}</div>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-500 mb-1">Email Address</label>
                                <div className="text-gray-900 font-medium">{user.email}</div>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-500 mb-1">Role</label>
                                <span className={`inline-flex px-3 py-1 text-xs font-semibold rounded-full ${user.role === 'ADMIN' ? 'bg-red-100 text-red-800' : 'bg-blue-100 text-blue-800'
                                    }`}>
                                    {user.role}
                                </span>
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-500 mb-1">Phone</label>
                                <div className="text-gray-900 font-medium">{user.phone || 'N/A'}</div>
                            </div>
                        </div>
                    </div>
                </div>

                <div className="bg-white rounded-lg shadow-sm border border-gray-200 overflow-hidden">
                    <div className="px-6 py-5 border-b border-gray-200 bg-gray-50 flex justify-between items-center">
                        <h3 className="text-lg font-medium text-gray-900">Attendance History</h3>
                        <span className="text-sm text-gray-500">{attendance.length} records found</span>
                    </div>
                    <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50">
                                <tr>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Date & Time</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Event</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Confidence</th>
                                </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                                {attendance.length > 0 ? (
                                    attendance.map((record) => (
                                        <tr key={record.id}>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                                                {new Date(record.timestamp).toLocaleString()}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap">
                                                <span className={`px-2 py-1 text-xs font-semibold rounded-full ${record.eventType === 'ENTRY'
                                                        ? 'bg-green-100 text-green-800'
                                                        : 'bg-orange-100 text-orange-800'
                                                    }`}>
                                                    {record.eventType}
                                                </span>
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                                                {(record.confidence * 100).toFixed(1)}%
                                            </td>
                                        </tr>
                                    ))
                                ) : (
                                    <tr>
                                        <td colSpan="3" className="px-6 py-8 text-center text-gray-500">
                                            No attendance records found for this user.
                                        </td>
                                    </tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
        </DashboardLayout>
    );
};
