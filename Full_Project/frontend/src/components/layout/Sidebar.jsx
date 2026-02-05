import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';

export const Sidebar = ({ isAdmin }) => {
    const { logout } = useAuth();
    const navigate = useNavigate();
    const location = useLocation();

    const handleLogout = () => {
        logout();
        navigate('/');
    };

    const userLinks = [
        { path: '/user/dashboard', label: 'Dashboard', icon: '📊' },
        { path: '/user/profile', label: 'Profile', icon: '👤' },
        { path: '/user/assets', label: 'Assets', icon: '💼' },
        { path: '/user/leaves', label: 'Leave', icon: '📅' },
        { path: '/user/status', label: 'Status', icon: '📋' },
        { path: '/user/attendance', label: 'Attendance', icon: '⏰' },
    ];

    const adminLinks = [
        { path: '/admin/dashboard', label: 'Dashboard', icon: '📊' },
        { path: '/admin/users', label: 'Users', icon: '👥' },
        { path: '/admin/assets', label: 'Assets', icon: '💼' },
        { path: '/admin/leaves', label: 'Leaves', icon: '📅' },
        { path: '/admin/attendance', label: 'Attendance', icon: '⏰' },
    ];

    const links = isAdmin ? adminLinks : userLinks;

    return (
        <div className="w-64 bg-gray-900 text-white min-h-screen flex flex-col">
            <div className="p-6 border-b border-gray-700">
                <h1 className="text-2xl font-bold">Attendance System</h1>
                <p className="text-sm text-gray-400 mt-1">{isAdmin ? 'Admin Panel' : 'User Portal'}</p>
            </div>

            <nav className="flex-1 p-4 space-y-2">
                {links.map((link) => (
                    <Link
                        key={link.path}
                        to={link.path}
                        className={`flex items-center space-x-3 px-4 py-3 rounded-lg transition ${location.pathname === link.path
                                ? 'bg-primary-600 text-white'
                                : 'text-gray-300 hover:bg-gray-800'
                            }`}
                    >
                        <span className="text-xl">{link.icon}</span>
                        <span className="font-medium">{link.label}</span>
                    </Link>
                ))}
            </nav>

            <div className="p-4 border-t border-gray-700">
                <button
                    onClick={handleLogout}
                    className="w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-gray-300 hover:bg-red-600 hover:text-white transition"
                >
                    <span className="text-xl">🚪</span>
                    <span className="font-medium">Logout</span>
                </button>
            </div>
        </div>
    );
};
