import { useState, useEffect } from 'react';
import { attendanceAPI } from '../services/api';

function StatsCard({ title, value, icon, color, subtitle }) {
    return (
        <div className="glass-card p-6">
            <div className="flex items-start justify-between">
                <div>
                    <p className="text-dark-400 text-sm font-medium mb-1">{title}</p>
                    <p className={`text-3xl font-bold ${color}`}>{value}</p>
                    {subtitle && <p className="text-dark-500 text-xs mt-1">{subtitle}</p>}
                </div>
                <div className="text-3xl opacity-60">{icon}</div>
            </div>
        </div>
    );
}

function formatTime(dateStr) {
    if (!dateStr) return '—';
    return new Date(dateStr).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

export default function Dashboard() {
    const [stats, setStats] = useState(null);
    const [loading, setLoading] = useState(true);
    const [time, setTime] = useState(new Date());

    useEffect(() => {
        const timer = setInterval(() => setTime(new Date()), 1000);
        return () => clearInterval(timer);
    }, []);

    useEffect(() => {
        loadStats();
        const interval = setInterval(loadStats, 30000); // refresh every 30s
        return () => clearInterval(interval);
    }, []);

    const loadStats = async () => {
        try {
            const res = await attendanceAPI.getDashboard();
            setStats(res.data);
        } catch (e) { console.error('Dashboard load error:', e); }
        finally { setLoading(false); }
    };

    if (loading) return <div className="flex items-center justify-center h-64"><div className="animate-spin rounded-full h-10 w-10 border-t-2 border-primary-500"></div></div>;

    return (
        <div className="space-y-8">
            {/* Header */}
            <div className="flex items-end justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-1">Dashboard</h1>
                    <p className="text-dark-400">Real-time attendance monitoring</p>
                </div>
                <div className="text-right">
                    <p className="text-2xl font-bold text-white font-mono">
                        {time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                    </p>
                    <p className="text-dark-400 text-sm">{time.toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' })}</p>
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
                <StatsCard title="Total Employees" value={stats?.totalEmployees ?? 0} icon="👥" color="text-primary-400" subtitle="Active workforce" />
                <StatsCard title="Present Today" value={stats?.presentToday ?? 0} icon="✅" color="text-accent-emerald" subtitle="Checked in" />
                <StatsCard title="Absent Today" value={stats?.absentToday ?? 0} icon="❌" color="text-accent-rose" subtitle="Not yet arrived" />
                <StatsCard title="Late Arrivals" value={stats?.lateEmployees ?? 0} icon="⏰" color="text-accent-amber" subtitle="After 9:00 AM" />
            </div>

            {/* Recent Attendance */}
            <div className="glass-card overflow-hidden">
                <div className="p-6 border-b border-white/5">
                    <div className="flex items-center justify-between">
                        <h2 className="text-lg font-semibold text-white">Recent Attendance Logs</h2>
                        <span className="badge badge-info">Live</span>
                    </div>
                </div>
                <div className="overflow-x-auto">
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Employee</th>
                                <th>Entry Time</th>
                                <th>Exit Time</th>
                                <th>Confidence</th>
                                <th>Camera</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {(!stats?.recentLogs || stats.recentLogs.length === 0) ? (
                                <tr><td colSpan={6} className="text-center text-dark-500 py-8">No attendance records today</td></tr>
                            ) : (
                                stats.recentLogs.map((log) => (
                                    <tr key={log.id} className="animate-slide-in">
                                        <td>
                                            <div className="flex items-center gap-3">
                                                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${log.isUnknown ? 'bg-accent-amber/20 text-accent-amber' : 'bg-primary-600/20 text-primary-400'}`}>
                                                    {log.isUnknown ? '?' : log.employeeName?.[0]?.toUpperCase()}
                                                </div>
                                                <span className={log.isUnknown ? 'text-accent-amber' : 'text-dark-200'}>{log.employeeName || 'Unknown'}</span>
                                            </div>
                                        </td>
                                        <td className="text-dark-300">{formatTime(log.entryTime)}</td>
                                        <td className="text-dark-300">{formatTime(log.exitTime)}</td>
                                        <td><span className={`badge ${log.confidence > 0.7 ? 'badge-success' : log.confidence > 0.4 ? 'badge-warning' : 'badge-danger'}`}>{(log.confidence * 100).toFixed(0)}%</span></td>
                                        <td className="text-dark-400">{log.cameraId || '—'}</td>
                                        <td>
                                            {log.isUnknown ? <span className="badge badge-warning">Unknown</span> :
                                                log.exitTime ? <span className="badge badge-info">Completed</span> :
                                                    <span className="badge badge-success">Present</span>}
                                        </td>
                                    </tr>
                                ))
                            )}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
