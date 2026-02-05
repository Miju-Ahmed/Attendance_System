import { DashboardLayout } from '../layout/DashboardLayout';
import { useAuth } from '../../context/AuthContext';

export const UserDashboard = () => {
    const { user } = useAuth();

    return (
        <DashboardLayout>
            <div className="space-y-6">
                <div>
                    <h1 className="text-3xl font-bold text-gray-900">Welcome, {user?.name}!</h1>
                    <p className="text-gray-600 mt-2">Here's your dashboard overview</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <div className="card bg-gradient-to-br from-blue-500 to-blue-600 text-white">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-blue-100 text-sm">Profile Status</p>
                                <p className="text-2xl font-bold mt-1">Active</p>
                            </div>
                            <div className="text-4xl">👤</div>
                        </div>
                    </div>

                    <div className="card bg-gradient-to-br from-green-500 to-green-600 text-white">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-green-100 text-sm">My Assets</p>
                                <p className="text-2xl font-bold mt-1">View</p>
                            </div>
                            <div className="text-4xl">💼</div>
                        </div>
                    </div>

                    <div className="card bg-gradient-to-br from-purple-500 to-purple-600 text-white">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-purple-100 text-sm">Leave Status</p>
                                <p className="text-2xl font-bold mt-1">Check</p>
                            </div>
                            <div className="text-4xl">📅</div>
                        </div>
                    </div>

                    <div className="card bg-gradient-to-br from-orange-500 to-orange-600 text-white">
                        <div className="flex items-center justify-between">
                            <div>
                                <p className="text-orange-100 text-sm">Attendance</p>
                                <p className="text-2xl font-bold mt-1">Track</p>
                            </div>
                            <div className="text-4xl">⏰</div>
                        </div>
                    </div>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="card">
                        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
                        <div className="space-y-3">
                            <a href="/user/profile" className="block p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition">
                                <div className="flex items-center space-x-3">
                                    <span className="text-2xl">📸</span>
                                    <div>
                                        <p className="font-medium text-gray-900">Update Profile Photo</p>
                                        <p className="text-sm text-gray-500">Upload your profile picture</p>
                                    </div>
                                </div>
                            </a>
                            <a href="/user/assets" className="block p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition">
                                <div className="flex items-center space-x-3">
                                    <span className="text-2xl">📝</span>
                                    <div>
                                        <p className="font-medium text-gray-900">Request Asset</p>
                                        <p className="text-sm text-gray-500">Apply for new assets</p>
                                    </div>
                                </div>
                            </a>
                            <a href="/user/leaves" className="block p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition">
                                <div className="flex items-center space-x-3">
                                    <span className="text-2xl">✈️</span>
                                    <div>
                                        <p className="font-medium text-gray-900">Apply for Leave</p>
                                        <p className="text-sm text-gray-500">Submit leave application</p>
                                    </div>
                                </div>
                            </a>
                        </div>
                    </div>

                    <div className="card">
                        <h3 className="text-lg font-semibold text-gray-900 mb-4">User Information</h3>
                        <div className="space-y-3">
                            <div className="flex justify-between py-2 border-b border-gray-200">
                                <span className="text-gray-600">Name:</span>
                                <span className="font-medium text-gray-900">{user?.name}</span>
                            </div>
                            <div className="flex justify-between py-2 border-b border-gray-200">
                                <span className="text-gray-600">Email:</span>
                                <span className="font-medium text-gray-900">{user?.email}</span>
                            </div>
                            <div className="flex justify-between py-2 border-b border-gray-200">
                                <span className="text-gray-600">Role:</span>
                                <span className="px-3 py-1 bg-blue-100 text-blue-800 rounded-full text-sm font-medium">
                                    {user?.role}
                                </span>
                            </div>
                            <div className="flex justify-between py-2">
                                <span className="text-gray-600">User ID:</span>
                                <span className="font-medium text-gray-900">#{user?.id}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </DashboardLayout>
    );
};
