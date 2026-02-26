import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

export default function Login() {
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [error, setError] = useState('');
    const [loading, setLoading] = useState(false);
    const { login } = useAuth();
    const navigate = useNavigate();

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError('');
        try {
            await login(username, password);
            navigate('/');
        } catch (err) {
            setError(err.response?.data?.message || 'Login failed. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-screen flex items-center justify-center p-4 relative overflow-hidden">
            {/* Ambient background orbs */}
            <div className="absolute top-20 left-20 w-72 h-72 bg-primary-600/20 rounded-full blur-3xl"></div>
            <div className="absolute bottom-20 right-20 w-96 h-96 bg-accent-cyan/10 rounded-full blur-3xl"></div>
            <div className="absolute top-1/2 left-1/2 w-64 h-64 bg-accent-rose/10 rounded-full blur-3xl -translate-x-1/2 -translate-y-1/2"></div>

            <div className="glass-card p-10 w-full max-w-md animate-fade-in relative z-10">
                {/* Logo */}
                <div className="text-center mb-8">
                    <div className="w-16 h-16 mx-auto rounded-2xl bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center text-white text-2xl font-bold mb-4 animate-pulse-glow">
                        AI
                    </div>
                    <h1 className="text-2xl font-bold text-white mb-1">Welcome Back</h1>
                    <p className="text-dark-400 text-sm">Sign in to your attendance dashboard</p>
                </div>

                {error && (
                    <div className="mb-5 p-3 rounded-xl bg-accent-rose/10 border border-accent-rose/20 text-accent-rose text-sm text-center animate-fade-in">
                        {error}
                    </div>
                )}

                <form onSubmit={handleSubmit} className="space-y-5">
                    <div>
                        <label className="block text-sm font-medium text-dark-300 mb-2">Username</label>
                        <input
                            type="text" value={username} onChange={(e) => setUsername(e.target.value)}
                            placeholder="Enter username" required autoFocus
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-dark-300 mb-2">Password</label>
                        <input
                            type="password" value={password} onChange={(e) => setPassword(e.target.value)}
                            placeholder="Enter password" required
                        />
                    </div>
                    <button type="submit" disabled={loading}
                        className="btn-primary w-full py-3 text-base flex items-center justify-center gap-2">
                        {loading ? (
                            <><div className="animate-spin rounded-full h-5 w-5 border-t-2 border-white"></div> Signing in...</>
                        ) : 'Sign In'}
                    </button>
                </form>

                <p className="text-center text-dark-500 text-xs mt-6">
                    Default: <span className="text-dark-300">admin</span> / <span className="text-dark-300">Admin@123</span>
                </p>
            </div>
        </div>
    );
}
