import { useState } from 'react';
import { attendanceAPI } from '../services/api';

function formatTime(dateStr) {
    if (!dateStr) return '—';
    return new Date(dateStr).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
}
function formatDate(dateStr) {
    return new Date(dateStr).toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}

export default function Reports() {
    const [records, setRecords] = useState([]);
    const [loading, setLoading] = useState(false);
    const [startDate, setStartDate] = useState(new Date().toISOString().split('T')[0]);
    const [endDate, setEndDate] = useState(new Date().toISOString().split('T')[0]);
    const [department, setDepartment] = useState('');

    const fetchReport = async () => {
        setLoading(true);
        try {
            const res = await attendanceAPI.getReport({ startDate, endDate, department: department || undefined });
            setRecords(res.data);
        } catch (e) { console.error(e); }
        finally { setLoading(false); }
    };

    const exportCSV = () => {
        if (records.length === 0) return;
        const headers = ['Employee', 'Date', 'Entry', 'Exit', 'Confidence', 'Camera'];
        const rows = records.map(r => [
            r.employeeName, formatDate(r.date), formatTime(r.entryTime),
            formatTime(r.exitTime), (r.confidence * 100).toFixed(0) + '%', r.cameraId || ''
        ]);
        const csv = [headers, ...rows].map(r => r.join(',')).join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `attendance_report_${startDate}_${endDate}.csv`;
        a.click();
        URL.revokeObjectURL(url);
    };

    return (
        <div className="space-y-6">
            <div>
                <h1 className="text-3xl font-bold text-white mb-1">Attendance Reports</h1>
                <p className="text-dark-400">Generate and export attendance reports</p>
            </div>

            {/* Filters */}
            <div className="glass-card p-6">
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
                    <div>
                        <label className="block text-sm font-medium text-dark-300 mb-2">Start Date</label>
                        <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-dark-300 mb-2">End Date</label>
                        <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-dark-300 mb-2">Department</label>
                        <input value={department} onChange={e => setDepartment(e.target.value)} placeholder="All departments" />
                    </div>
                    <div className="flex gap-2">
                        <button onClick={fetchReport} disabled={loading} className="btn-primary flex-1">
                            {loading ? 'Loading...' : '📊 Generate'}
                        </button>
                        <button onClick={exportCSV} disabled={records.length === 0} className="btn-secondary">📥 CSV</button>
                    </div>
                </div>
            </div>

            {/* Summary */}
            {records.length > 0 && (
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="glass-card p-5 text-center">
                        <p className="text-dark-400 text-sm">Total Records</p>
                        <p className="text-2xl font-bold text-primary-400">{records.length}</p>
                    </div>
                    <div className="glass-card p-5 text-center">
                        <p className="text-dark-400 text-sm">Unique Employees</p>
                        <p className="text-2xl font-bold text-accent-emerald">
                            {new Set(records.filter(r => !r.isUnknown).map(r => r.employeeId)).size}
                        </p>
                    </div>
                    <div className="glass-card p-5 text-center">
                        <p className="text-dark-400 text-sm">Avg Confidence</p>
                        <p className="text-2xl font-bold text-accent-cyan">
                            {records.length > 0 ? (records.reduce((s, r) => s + r.confidence, 0) / records.length * 100).toFixed(0) + '%' : '—'}
                        </p>
                    </div>
                </div>
            )}

            {/* Results */}
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
                            </tr>
                        </thead>
                        <tbody>
                            {records.length === 0 ? (
                                <tr><td colSpan={6} className="text-center text-dark-500 py-8">Select a date range and click "Generate" to see reports</td></tr>
                            ) : records.map((r) => (
                                <tr key={r.id}>
                                    <td className="text-dark-200">{r.employeeName || 'Unknown'}</td>
                                    <td className="text-dark-300 text-sm">{formatDate(r.date)}</td>
                                    <td className="text-accent-emerald font-mono text-sm">{formatTime(r.entryTime)}</td>
                                    <td className="text-accent-rose font-mono text-sm">{formatTime(r.exitTime)}</td>
                                    <td className="text-dark-400 text-sm">{(r.confidence * 100).toFixed(0)}%</td>
                                    <td className="text-dark-400 text-sm">{r.cameraId || '—'}</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    );
}
