import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const navItems = [
    { path: '/', label: 'Dashboard', icon: '📊' },
    { path: '/employees', label: 'Employees', icon: '👥' },
    { path: '/attendance', label: 'Attendance', icon: '📋' },
    { path: '/reports', label: 'Reports', icon: '📈' },
    { path: '/cameras', label: 'Cameras', icon: '📹' },
];

export default function Layout() {
    const { user, logout } = useAuth();
    const navigate = useNavigate();

    const handleLogout = () => { logout(); navigate('/login'); };

    return (
        <div className="flex min-h-screen">
            {/* Sidebar */}
            <aside className="w-64 glass border-r border-white/5 flex flex-col fixed h-full z-20">
                {/* Brand */}
                <div className="p-6 border-b border-white/5">
                    <div className="flex items-center gap-3">
                        <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center text-white text-lg font-bold animate-pulse-glow">
                            AI
                        </div>
                        <div>
                            <h1 className="text-base font-bold text-white">Attendance</h1>
                            <p className="text-xs text-dark-400">AI-Powered System</p>
                        </div>
                    </div>
                </div>

                {/* Nav */}
                <nav className="flex-1 px-3 py-4 space-y-1">
                    {navItems.map((item) => (
                        <NavLink
                            key={item.path}
                            to={item.path}
                            end={item.path === '/'}
                            className={({ isActive }) =>
                                `flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 ${isActive
                                    ? 'bg-gradient-to-r from-primary-600/30 to-primary-500/10 text-primary-300 border border-primary-500/20'
                                    : 'text-dark-400 hover:text-dark-200 hover:bg-white/5'
                                }`
                            }
                        >
                            <span className="text-lg">{item.icon}</span>
                            {item.label}
                        </NavLink>
                    ))}
                </nav>

                {/* User */}
                <div className="p-4 border-t border-white/5">
                    <div className="flex items-center gap-3 mb-3">
                        <div className="w-9 h-9 rounded-full bg-gradient-to-br from-accent-cyan to-accent-emerald flex items-center justify-center text-white text-sm font-bold">
                            {user?.username?.[0]?.toUpperCase() || 'A'}
                        </div>
                        <div className="flex-1 min-w-0">
                            <p className="text-sm font-medium text-dark-200 truncate">{user?.username}</p>
                            <p className="text-xs text-dark-500">{user?.role}</p>
                        </div>
                    </div>
                    <button onClick={handleLogout}
                        className="w-full text-xs text-dark-400 hover:text-accent-rose transition py-2 rounded-lg hover:bg-accent-rose/10">
                        Sign Out
                    </button>
                </div>
            </aside>

            {/* Main Content */}
            <main className="flex-1 ml-64 p-8 min-h-screen">
                <div className="animate-fade-in">
                    <Outlet />
                </div>
            </main>
        </div>
    );
}
