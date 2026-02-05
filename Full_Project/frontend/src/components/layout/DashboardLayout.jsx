import { Sidebar } from './Sidebar';
import { useAuth } from '../../context/AuthContext';

export const DashboardLayout = ({ children }) => {
    const { user, isAdmin } = useAuth();

    return (
        <div className="flex min-h-screen bg-gray-50">
            <Sidebar isAdmin={isAdmin} />
            <div className="flex-1">
                <header className="bg-white shadow-sm border-b border-gray-200 px-8 py-4">
                    <div className="flex justify-between items-center">
                        <h2 className="text-2xl font-bold text-gray-800">
                            {isAdmin ? 'Admin Dashboard' : 'User Dashboard'}
                        </h2>
                        <div className="flex items-center space-x-4">
                            <div className="text-right">
                                <p className="text-sm font-medium text-gray-900">{user?.name}</p>
                                <p className="text-xs text-gray-500">{user?.email}</p>
                            </div>
                            <div className="h-10 w-10 rounded-full bg-primary-600 flex items-center justify-center text-white font-semibold">
                                {user?.name?.charAt(0).toUpperCase()}
                            </div>
                        </div>
                    </div>
                </header>
                <main className="p-8">{children}</main>
            </div>
        </div>
    );
};
