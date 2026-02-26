import { useState, useEffect } from 'react';
import { attendanceAPI } from '../services/api';

function formatTime(dateStr) {
    if (!dateStr) return '—';
    return new Date(dateStr).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}

function formatDate(dateStr) {
    return new Date(dateStr).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}

export default function Attendance() {
    const [records, setRecords] = useState([]);
    const [loading, setLoading] = useState(true);
    const [search, setSearch] = useState('');

    useEffect(() => { loadToday(); }, []);

    const loadToday = async () => {
        try { const res = await attendanceAPI.getToday(); setRecords(res.data); }
        catch (e) { console.error(e); }
        finally { setLoading(false); }
    };

    const filtered = records.filter(r =>
        !search || r.employeeName?.toLowerCase().includes(search.toLowerCase()) ||
        r.cameraId?.toLowerCase().includes(search.toLowerCase())
    );

    if (loading) return <div className="flex items-center justify-center h-64"><div className="animate-spin rounded-full h-10 w-10 border-t-2 border-primary-500"></div></div>;

    return (
        <div className="space-y-6">
            <div className="flex items-end justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-1">Attendance Logs</h1>
                    <p className="text-dark-400">Today's attendance records — {filtered.length} entries</p>
                </div>
                <button onClick={loadToday} className="btn-secondary">🔄 Refresh</button>
            </div>

            {/* Search */}
            <div className="max-w-md">
                <input value={search} onChange={e => setSearch(e.target.value)}
                    placeholder="🔍 Search by name or camera..." className="!rounded-full !pl-5" />
            </div>

            {/* Table */}
            <div className="glass-card overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>Employee</th>
                                <th>Date</th>
                                <th>Entry Time</th>
                                <th>Exit Time</th>
                                <th>Confidence</th>
                                <th>Camera</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {filtered.length === 0 ? (
                                <tr><td colSpan={7} className="text-center text-dark-500 py-8">No attendance records found</td></tr>
                            ) : filtered.map((r) => (
                                <tr key={r.id}>
                                    <td>
                                        <div className="flex items-center gap-3">
                                            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold ${r.isUnknown ? 'bg-accent-amber/20 text-accent-amber' : 'bg-primary-600/20 text-primary-400'}`}>
                                                {r.isUnknown ? '?' : r.employeeName?.[0]?.toUpperCase()}
                                            </div>
                                            <span className={r.isUnknown ? 'text-accent-amber' : 'text-dark-200'}>{r.employeeName || 'Unknown'}</span>
                                        </div>
                                    </td>
                                    <td className="text-dark-300 text-sm">{formatDate(r.date)}</td>
                                    <td className="text-accent-emerald font-mono text-sm">{formatTime(r.entryTime)}</td>
                                    <td className="text-accent-rose font-mono text-sm">{formatTime(r.exitTime)}</td>
                                    <td>
                                        <div className="flex items-center gap-2">
                                            <div className="w-16 h-1.5 bg-dark-800 rounded-full overflow-hidden">
                                                <div className={`h-full rounded-full ${r.confidence > 0.7 ? 'bg-accent-emerald' : r.confidence > 0.4 ? 'bg-accent-amber' : 'bg-accent-rose'}`}
                                                    style={{ width: `${r.confidence * 100}%` }}></div>
                                            </div>
                                            <span className="text-dark-400 text-xs">{(r.confidence * 100).toFixed(0)}%</span>
                                        </div>
                                    </td>
                                    <td className="text-dark-400 text-sm">{r.cameraId || '—'}</td>
                                    <td>
                                        {r.isUnknown ? <span className="badge badge-warning">Unknown</span> :
                                            r.exitTime ? <span className="badge badge-info">Completed</span> :
                                                <span className="badge badge-success">Present</span>}
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
