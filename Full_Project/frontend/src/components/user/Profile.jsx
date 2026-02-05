import { useState, useEffect } from 'react';
import { DashboardLayout } from '../layout/DashboardLayout';
import { useAuth } from '../../context/AuthContext';
import { userService } from '../../services/api';

export const Profile = () => {
    const { user } = useAuth();
    const [formData, setFormData] = useState({ name: '', phone: '', address: '' });
    const [file, setFile] = useState(null);
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState('');

    useEffect(() => {
        if (user) {
            setFormData({
                name: user.name || '',
                phone: user.phone || '',
                address: user.address || '',
            });
        }
    }, [user]);

    const handleChange = (e) => {
        setFormData({ ...formData, [e.target.name]: e.target.value });
    };

    const handleFileChange = (e) => {
        setFile(e.target.files[0]);
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setMessage('');

        try {
            await userService.updateUser(user.id, formData);
            setMessage('Profile updated successfully!');
        } catch (error) {
            setMessage('Failed to update profile');
        } finally {
            setLoading(false);
        }
    };

    const handlePhotoUpload = async (e) => {
        e.preventDefault();
        if (!file) return;

        setLoading(true);
        setMessage('');

        try {
            await userService.uploadProfilePhoto(user.id, file);
            setMessage('Photo uploaded successfully!');
            setFile(null);
        } catch (error) {
            setMessage('Failed to upload photo');
        } finally {
            setLoading(false);
        }
    };

    return (
        <DashboardLayout>
            <div className="max-w-4xl">
                <h1 className="text-3xl font-bold text-gray-900 mb-6">My Profile</h1>

                {message && (
                    <div className={`mb-4 p-4 rounded-lg ${message.includes('success') ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                        {message}
                    </div>
                )}

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div className="card">
                        <h2 className="text-xl font-semibold text-gray-900 mb-4">Profile Photo</h2>
                        <div className="text-center">
                            <div className="mb-4">
                                <div className="h-32 w-32 mx-auto rounded-full bg-primary-600 flex items-center justify-center text-white text-4xl font-bold">
                                    {user?.name?.charAt(0).toUpperCase()}
                                </div>
                            </div>
                            <form onSubmit={handlePhotoUpload} className="space-y-4">
                                <input
                                    type="file"
                                    accept="image/*"
                                    onChange={handleFileChange}
                                    className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-semibold file:bg-primary-50 file:text-primary-700 hover:file:bg-primary-100"
                                />
                                <button
                                    type="submit"
                                    disabled={!file || loading}
                                    className="btn-primary w-full disabled:opacity-50"
                                >
                                    {loading ? 'Uploading...' : 'Upload Photo'}
                                </button>
                            </form>
                        </div>
                    </div>

                    <div className="card">
                        <h2 className="text-xl font-semibold text-gray-900 mb-4">Update Information</h2>
                        <form onSubmit={handleSubmit} className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">Name</label>
                                <input
                                    type="text"
                                    name="name"
                                    value={formData.name}
                                    onChange={handleChange}
                                    className="input-field"
                                    required
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">Phone</label>
                                <input
                                    type="tel"
                                    name="phone"
                                    value={formData.phone}
                                    onChange={handleChange}
                                    className="input-field"
                                />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-gray-700 mb-2">Address</label>
                                <textarea
                                    name="address"
                                    value={formData.address}
                                    onChange={handleChange}
                                    className="input-field"
                                    rows={3}
                                />
                            </div>
                            <button type="submit" disabled={loading} className="btn-primary w-full disabled:opacity-50">
                                {loading ? 'Updating...' : 'Update Profile'}
                            </button>
                        </form>
                    </div>
                </div>
            </div>
        </DashboardLayout>
    );
};
