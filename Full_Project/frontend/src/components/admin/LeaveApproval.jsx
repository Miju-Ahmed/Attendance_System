import { useState, useEffect } from 'react';
import { DashboardLayout } from '../layout/DashboardLayout';
import { leaveService } from '../../services/api';

export const LeaveApproval = () => {
    const [leaves, setLeaves] = useState([]);
    const [loading, setLoading] = useState(true);
    const [message, setMessage] = useState('');

    useEffect(() => {
        fetchLeaves();
    }, []);

    const fetchLeaves = async () => {
        try {
            const data = await leaveService.getPendingLeaves();
            setLeaves(data);
        } catch (error) {
            console.error('Failed to fetch leaves');
        } finally {
            setLoading(false);
        }
    };

    const handleApprove = async (id) => {
        try {
            await leaveService.approveLeave(id);
            setMessage('Leave approved successfully');
            fetchLeaves();
        } catch (error) {
            setMessage('Failed to approve leave');
        }
    };

    const handleReject = async (id) => {
        try {
            await leaveService.rejectLeave(id);
            setMessage('Leave rejected');
            fetchLeaves();
        } catch (error) {
            setMessage('Failed to reject leave');
        }
    };

    if (loading) {
        return (
            <DashboardLayout>
                <div className="text-center py-12">Loading...</div>
            </DashboardLayout>
        );
    }

    return (
        <DashboardLayout>
            <div>
                <h1 className="text-3xl font-bold text-gray-900 mb-6">Leave Approval</h1>

                {message && (
                    <div className={`mb-4 p-4 rounded-lg ${message.includes('success') || message.includes('approved') ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                        {message}
                    </div>
                )}

                <div className="card">
                    <h2 className="text-xl font-semibold text-gray-900 mb-4">Pending Leave Requests</h2>
                    {leaves.length === 0 ? (
                        <p className="text-gray-500">No pending leave requests</p>
                    ) : (
                        <div className="overflow-x-auto">
                            <table className="min-w-full divide-y divide-gray-200">
                                <thead className="bg-gray-50">
                                    <tr>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Employee</th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Start Date</th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">End Date</th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Reason</th>
                                        <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
                                    </tr>
                                </thead>
                                <tbody className="bg-white divide-y divide-gray-200">
                                    {leaves.map((leave) => (
                                        <tr key={leave.id}>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                                                {leave.user?.name || 'Unknown'}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{leave.leaveType}</td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{leave.startDate}</td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{leave.endDate}</td>
                                            <td className="px-6 py-4 text-sm text-gray-500 max-w-xs truncate">
                                                {leave.reason || 'No reason provided'}
                                            </td>
                                            <td className="px-6 py-4 whitespace-nowrap text-sm space-x-2">
                                                <button
                                                    onClick={() => handleApprove(leave.id)}
                                                    className="px-3 py-1 bg-green-600 text-white rounded hover:bg-green-700"
                                                >
                                                    Approve
                                                </button>
                                                <button
                                                    onClick={() => handleReject(leave.id)}
                                                    className="px-3 py-1 bg-red-600 text-white rounded hover:bg-red-700"
                                                >
                                                    Reject
                                                </button>
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
