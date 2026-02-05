import { useState, useEffect } from 'react';
import { DashboardLayout } from '../layout/DashboardLayout';
import { assetService } from '../../services/api';

export const AssetApplication = () => {
    const [assets, setAssets] = useState([]);
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState('');

    useEffect(() => {
        fetchAssets();
    }, []);

    const fetchAssets = async () => {
        try {
            const data = await assetService.getAllAssets();
            setAssets(data);
        } catch (error) {
            console.error('Failed to fetch assets');
        }
    };

    const handleRequest = async (assetId) => {
        setLoading(true);
        setMessage('');

        try {
            await assetService.requestAsset(assetId);
            setMessage('Asset request submitted successfully!');
            fetchAssets();
        } catch (error) {
            setMessage('Failed to request asset');
        } finally {
            setLoading(false);
        }
    };

    return (
        <DashboardLayout>
            <div>
                <h1 className="text-3xl font-bold text-gray-900 mb-6">Asset Application</h1>

                {message && (
                    <div className={`mb-4 p-4 rounded-lg ${message.includes('success') ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'}`}>
                        {message}
                    </div>
                )}

                <div className="card">
                    <h2 className="text-xl font-semibold text-gray-900 mb-4">Available Assets</h2>
                    <div className="overflow-x-auto">
                        <table className="min-w-full divide-y divide-gray-200">
                            <thead className="bg-gray-50">
                                <tr>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Asset Name</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Type</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Action</th>
                                </tr>
                            </thead>
                            <tbody className="bg-white divide-y divide-gray-200">
                                {assets.map((asset) => (
                                    <tr key={asset.id}>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                                            {asset.assetName}
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{asset.assetType}</td>
                                        <td className="px-6 py-4 whitespace-nowrap">
                                            <span className={`px-2 py-1 text-xs font-semibold rounded-full ${asset.status === 'AVAILABLE' ? 'bg-green-100 text-green-800' :
                                                    asset.status === 'ASSIGNED' ? 'bg-blue-100 text-blue-800' :
                                                        'bg-yellow-100 text-yellow-800'
                                                }`}>
                                                {asset.status}
                                            </span>
                                        </td>
                                        <td className="px-6 py-4 whitespace-nowrap text-sm">
                                            {asset.status === 'AVAILABLE' && (
                                                <button
                                                    onClick={() => handleRequest(asset.id)}
                                                    disabled={loading}
                                                    className="btn-primary text-sm disabled:opacity-50"
                                                >
                                                    Request
                                                </button>
                                            )}
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
