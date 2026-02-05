import { useState } from 'react';
import { DashboardLayout } from '../layout/DashboardLayout';
import { leaveService } from '../../services/api';

export const LeaveApplication = () => {
    const [formData, setFormData] = useState({
        leaveType: '',
        startDate: '',
        endDate: '',
        reason: '',
    });
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState('');

    const leaveTypes = ['Sick Leave', 'Casual Leave', 'Annual Leave', 'Emergency Leave'];

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setMessage('');

        try {
            await leaveService.applyLeave(formData);
            setMessage('Leave application submitted successfully!');
            setFormData({ leaveType: '', startDate: '', endDate: '', reason: '' });
        } catch (error) {
            setMessage('Failed to submit leave application');
        } finally {
            setLoading(false);
        }
    };

    return (
        <DashboardLayout>
            <div className="max-w-2xl">
                <h1 className="text-3xl font-bold text-gray-900 mb-6">Apply for Leave</h1>

                {message && (
                    <div className={`mb-4 p-4 rounded-lg ${message.includes('success') ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                        {message}
                    </div>
                )}

                <div className="card">
                    <form onSubmit={handleSubmit} className="space-y-6">
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">Leave Type</label>
                            <select
                                name="leaveType"
                                value={formData.leaveType}
                                onChange={handleChange}
                                className="input-field"
                                required
                            >
                                <option value="">Select leave type</option>
                                {leaveTypes.map((type) => (
                                    <option key={type} value={type}>
                                        {type}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <div className="grid grid-cols-2 gap-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">Start Date</label>
                                <input
                                    type="date"
                                    name="startDate"
                                    value={formData.startDate}
                                    onChange={handleChange}
                                    className="input-field"
                                    required
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">End Date</label>
                                <input
                                    type="date"
                                    name="endDate"
                                    value={formData.endDate}
                                    onChange={handleChange}
                                    className="input-field"
                                    required
                                />
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">Reason</label>
                            <textarea
                                name="reason"
                                value={formData.reason}
                                onChange={handleChange}
                                className="input-field"
                                rows={4}
                                placeholder="Please provide a reason for your leave..."
                            />
                        </div>

                        <button type="submit" disabled={loading} className="btn-primary w-full disabled:opacity-50">
                            {loading ? 'Submitting...' : 'Submit Application'}
                        </button>
                    </form>
                </div>
            </div>
        </DashboardLayout>
    );
};
