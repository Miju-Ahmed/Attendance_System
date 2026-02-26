import { useState, useEffect } from 'react';
import { employeeAPI } from '../services/api';

export default function Employees() {
    const [employees, setEmployees] = useState([]);
    const [loading, setLoading] = useState(true);
    const [showModal, setShowModal] = useState(false);
    const [editId, setEditId] = useState(null);
    const [form, setForm] = useState({ name: '', department: '', position: '' });

    useEffect(() => { loadEmployees(); }, []);

    const loadEmployees = async () => {
        try { const res = await employeeAPI.getAll(); setEmployees(res.data); }
        catch (e) { console.error(e); }
        finally { setLoading(false); }
    };

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            if (editId) { await employeeAPI.update(editId, form); }
            else { await employeeAPI.create(form); }
            setShowModal(false); resetForm(); loadEmployees();
        } catch (e) { alert(e.response?.data?.message || 'Error saving employee'); }
    };

    const handleDelete = async (id) => {
        if (!confirm('Delete this employee?')) return;
        try { await employeeAPI.delete(id); loadEmployees(); }
        catch (e) { alert('Error deleting employee'); }
    };

    const handleEdit = (emp) => {
        setEditId(emp.id);
        setForm({ name: emp.name, department: emp.department, position: emp.position });
        setShowModal(true);
    };

    const handleUploadFace = async (id) => {
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = 'image/*';
        input.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) return;
            const formData = new FormData();
            formData.append('file', file);
            try { await employeeAPI.uploadFace(id, formData); alert('Face image uploaded!'); loadEmployees(); }
            catch (err) { alert('Upload failed'); }
        };
        input.click();
    };

    const resetForm = () => { setForm({ name: '', department: '', position: '' }); setEditId(null); };

    if (loading) return <div className="flex items-center justify-center h-64"><div className="animate-spin rounded-full h-10 w-10 border-t-2 border-primary-500"></div></div>;

    return (
        <div className="space-y-6">
            <div className="flex items-end justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-1">Employee Management</h1>
                    <p className="text-dark-400">{employees.length} employees registered</p>
                </div>
                <button className="btn-primary" onClick={() => { resetForm(); setShowModal(true); }}>+ Add Employee</button>
            </div>

            <div className="glass-card overflow-hidden">
                <div className="overflow-x-auto">
                    <table className="data-table">
                        <thead>
                            <tr>
                                <th>ID</th>
                                <th>Name</th>
                                <th>Department</th>
                                <th>Position</th>
                                <th>Created</th>
                                <th>Status</th>
                                <th>Actions</th>
                            </tr>
                        </thead>
                        <tbody>
                            {employees.length === 0 ? (
                                <tr><td colSpan={7} className="text-center text-dark-500 py-8">No employees found. Click "Add Employee" to get started.</td></tr>
                            ) : employees.map((emp) => (
                                <tr key={emp.id}>
                                    <td className="text-dark-400 font-mono text-sm">#{emp.id}</td>
                                    <td>
                                        <div className="flex items-center gap-3">
                                            <div className="w-9 h-9 rounded-full bg-gradient-to-br from-primary-500 to-primary-700 flex items-center justify-center text-white text-sm font-bold">
                                                {emp.name[0]?.toUpperCase()}
                                            </div>
                                            <span className="text-dark-200 font-medium">{emp.name}</span>
                                        </div>
                                    </td>
                                    <td className="text-dark-300">{emp.department || '—'}</td>
                                    <td className="text-dark-300">{emp.position || '—'}</td>
                                    <td className="text-dark-400 text-sm">{new Date(emp.createdAt).toLocaleDateString()}</td>
                                    <td><span className={`badge ${emp.isActive ? 'badge-success' : 'badge-danger'}`}>{emp.isActive ? 'Active' : 'Inactive'}</span></td>
                                    <td>
                                        <div className="flex gap-2">
                                            <button onClick={() => handleEdit(emp)} className="btn-secondary text-xs py-1 px-3">Edit</button>
                                            <button onClick={() => handleUploadFace(emp.id)} className="btn-secondary text-xs py-1 px-3" title="Upload face image">📷</button>
                                            <button onClick={() => handleDelete(emp.id)} className="btn-danger text-xs py-1 px-3">Delete</button>
                                        </div>
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>

            {/* Modal */}
            {showModal && (
                <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in">
                    <div className="glass-card p-8 w-full max-w-md">
                        <h2 className="text-xl font-bold text-white mb-6">{editId ? 'Edit Employee' : 'Add New Employee'}</h2>
                        <form onSubmit={handleSubmit} className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-dark-300 mb-2">Full Name</label>
                                <input value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} required placeholder="e.g. John Doe" />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-dark-300 mb-2">Department</label>
                                <input value={form.department} onChange={e => setForm({ ...form, department: e.target.value })} placeholder="e.g. Engineering" />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-dark-300 mb-2">Position</label>
                                <input value={form.position} onChange={e => setForm({ ...form, position: e.target.value })} placeholder="e.g. Software Engineer" />
                            </div>
                            <div className="flex gap-3 pt-2">
                                <button type="submit" className="btn-primary flex-1">
                                    {editId ? 'Update' : 'Create'}
                                </button>
                                <button type="button" onClick={() => setShowModal(false)} className="btn-secondary flex-1">Cancel</button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </div>
    );
}
