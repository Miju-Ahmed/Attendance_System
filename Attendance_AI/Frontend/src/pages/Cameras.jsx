import { useState, useEffect, useRef, useCallback } from 'react';
import { cameraAPI } from '../services/api';

// ──────────────────────────────────────────────────────────────
// Utility: human-readable file size
// ──────────────────────────────────────────────────────────────
function formatSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

function formatDate(dateStr) {
    if (!dateStr) return '—';
    return new Date(dateStr).toLocaleString('en-US', {
        month: 'short', day: 'numeric', year: 'numeric',
        hour: '2-digit', minute: '2-digit',
    });
}

// ──────────────────────────────────────────────────────────────
// MAIN COMPONENT
// ──────────────────────────────────────────────────────────────
export default function Cameras() {
    const [activeTab, setActiveTab] = useState('cameras'); // 'cameras' | 'videos'

    // ── Camera state ──
    const [cameras, setCameras] = useState([]);
    const [loadingCam, setLoadingCam] = useState(true);
    const [showModal, setShowModal] = useState(false);
    const [editId, setEditId] = useState(null);
    const [form, setForm] = useState({ name: '', rtspUrl: '', location: '' });

    // ── Video state ──
    const [videos, setVideos] = useState([]);
    const [loadingVid, setLoadingVid] = useState(true);
    const [uploading, setUploading] = useState(false);
    const [uploadProgress, setUploadProgress] = useState(0);
    const [dragOver, setDragOver] = useState(false);
    const fileInputRef = useRef(null);

    // ── Load data ──
    useEffect(() => { loadCameras(); loadVideos(); }, []);

    const loadCameras = async () => {
        try { const res = await cameraAPI.getAll(); setCameras(res.data); }
        catch (e) { console.error(e); }
        finally { setLoadingCam(false); }
    };

    const loadVideos = async () => {
        try { const res = await cameraAPI.getVideos(); setVideos(res.data); }
        catch (e) { console.error(e); }
        finally { setLoadingVid(false); }
    };

    // ── Camera handlers ──
    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            if (editId) { await cameraAPI.update(editId, form); }
            else { await cameraAPI.create(form); }
            setShowModal(false); resetForm(); loadCameras();
        } catch (e) { alert(e.response?.data?.message || 'Error saving camera'); }
    };

    const handleDelete = async (id) => {
        if (!confirm('Delete this camera?')) return;
        try { await cameraAPI.delete(id); loadCameras(); }
        catch (e) { alert('Error deleting camera'); }
    };

    const handleEdit = (cam) => {
        setEditId(cam.id);
        setForm({ name: cam.name, rtspUrl: cam.rtspUrl, location: cam.location });
        setShowModal(true);
    };

    const resetForm = () => { setForm({ name: '', rtspUrl: '', location: '' }); setEditId(null); };

    // ── Video upload handlers ──
    const handleFileSelect = useCallback((file) => {
        if (!file) return;
        const allowed = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.webm'];
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (!allowed.includes(ext)) {
            alert(`Unsupported file type. Allowed: ${allowed.join(', ')}`);
            return;
        }
        uploadFile(file);
    }, []);

    const uploadFile = async (file) => {
        setUploading(true);
        setUploadProgress(0);
        const formData = new FormData();
        formData.append('file', file);
        try {
            await cameraAPI.uploadVideo(formData, (e) => {
                const pct = Math.round((e.loaded * 100) / e.total);
                setUploadProgress(pct);
            });
            loadVideos();
        } catch (e) {
            alert(e.response?.data?.message || 'Upload failed');
        } finally {
            setUploading(false);
            setUploadProgress(0);
        }
    };

    const handleDeleteVideo = async (fileName) => {
        if (!confirm('Delete this video?')) return;
        try { await cameraAPI.deleteVideo(fileName); loadVideos(); }
        catch (e) { alert('Error deleting video'); }
    };

    const onDrop = useCallback((e) => {
        e.preventDefault();
        setDragOver(false);
        const file = e.dataTransfer?.files?.[0];
        handleFileSelect(file);
    }, [handleFileSelect]);

    const onDragOver = (e) => { e.preventDefault(); setDragOver(true); };
    const onDragLeave = () => setDragOver(false);

    // ── Loading state ──
    const isLoading = activeTab === 'cameras' ? loadingCam : loadingVid;
    if (isLoading) return (
        <div className="flex items-center justify-center h-64">
            <div className="animate-spin rounded-full h-10 w-10 border-t-2 border-primary-500"></div>
        </div>
    );

    // ──────────────────────────────────────────────────────────────
    // RENDER
    // ──────────────────────────────────────────────────────────────
    return (
        <div className="space-y-6">
            {/* ── Header ── */}
            <div className="flex items-end justify-between">
                <div>
                    <h1 className="text-3xl font-bold text-white mb-1">Camera & Video Sources</h1>
                    <p className="text-dark-400">
                        {activeTab === 'cameras'
                            ? `${cameras.length} live cameras configured`
                            : `${videos.length} videos uploaded`
                        }
                    </p>
                </div>
                {activeTab === 'cameras' && (
                    <button className="btn-primary" onClick={() => { resetForm(); setShowModal(true); }}>+ Add Camera</button>
                )}
            </div>

            {/* ── Tab Bar ── */}
            <div className="tab-bar" style={{ maxWidth: 420 }}>
                <button
                    id="tab-cameras"
                    className={`tab-btn ${activeTab === 'cameras' ? 'active' : ''}`}
                    onClick={() => setActiveTab('cameras')}
                >
                    <span>📹</span> Live Cameras
                </button>
                <button
                    id="tab-videos"
                    className={`tab-btn ${activeTab === 'videos' ? 'active' : ''}`}
                    onClick={() => setActiveTab('videos')}
                >
                    <span>📤</span> Video Upload
                </button>
            </div>

            {/* ── Tab Content ── */}
            <div className="animate-fade-in">
                {activeTab === 'cameras' ? (
                    /* ========== LIVE CAMERAS TAB ========== */
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
                        {cameras.length === 0 ? (
                            <div className="col-span-3 glass-card p-12 text-center">
                                <p className="text-4xl mb-4">📹</p>
                                <p className="text-dark-400">No cameras configured. Click "Add Camera" to get started.</p>
                            </div>
                        ) : cameras.map((cam) => (
                            <div key={cam.id} className="glass-card p-6 space-y-4">
                                <div className="flex items-start justify-between">
                                    <div className="flex items-center gap-3">
                                        <div className={`w-10 h-10 rounded-xl flex items-center justify-center text-lg ${cam.isActive ? 'bg-accent-emerald/15 text-accent-emerald' : 'bg-dark-700 text-dark-500'}`}>
                                            📹
                                        </div>
                                        <div>
                                            <h3 className="text-white font-semibold">{cam.name}</h3>
                                            <p className="text-dark-400 text-sm">{cam.location || 'No location'}</p>
                                        </div>
                                    </div>
                                    <span className={`badge ${cam.isActive ? 'badge-success' : 'badge-danger'}`}>
                                        {cam.isActive ? 'Active' : 'Inactive'}
                                    </span>
                                </div>
                                <div className="p-3 rounded-lg bg-dark-900/40">
                                    <p className="text-xs text-dark-500 mb-1">RTSP URL</p>
                                    <p className="text-dark-300 text-sm font-mono break-all">{cam.rtspUrl}</p>
                                </div>
                                <div className="flex gap-2">
                                    <button onClick={() => handleEdit(cam)} className="btn-secondary text-xs flex-1">Edit</button>
                                    <button onClick={() => handleDelete(cam.id)} className="btn-danger text-xs flex-1">Delete</button>
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    /* ========== VIDEO UPLOAD TAB ========== */
                    <div className="space-y-6">
                        {/* Drop Zone */}
                        <div
                            className={`drop-zone ${dragOver ? 'drag-over' : ''}`}
                            onDrop={onDrop}
                            onDragOver={onDragOver}
                            onDragLeave={onDragLeave}
                            onClick={() => fileInputRef.current?.click()}
                        >
                            <input
                                ref={fileInputRef}
                                type="file"
                                accept=".mp4,.avi,.mkv,.mov,.wmv,.webm"
                                style={{ display: 'none' }}
                                onChange={(e) => handleFileSelect(e.target.files?.[0])}
                            />
                            {uploading ? (
                                <div className="space-y-4">
                                    <p className="text-4xl">⏳</p>
                                    <p className="text-white font-semibold text-lg">Uploading…</p>
                                    <div className="max-w-md mx-auto">
                                        <div className="progress-bar-track">
                                            <div
                                                className="progress-bar-fill"
                                                style={{ width: `${uploadProgress}%` }}
                                            />
                                        </div>
                                        <p className="text-primary-400 text-sm mt-2 font-mono">{uploadProgress}%</p>
                                    </div>
                                </div>
                            ) : (
                                <div className="space-y-3">
                                    <p className="text-5xl">📂</p>
                                    <p className="text-white font-semibold text-lg">Drag & drop a video file here</p>
                                    <p className="text-dark-400 text-sm">or click to browse</p>
                                    <p className="text-dark-500 text-xs mt-2">
                                        Supports MP4, AVI, MKV, MOV, WMV, WebM — up to 500 MB
                                    </p>
                                </div>
                            )}
                        </div>

                        {/* Uploaded Videos List */}
                        {videos.length > 0 && (
                            <div className="glass-card overflow-hidden">
                                <div className="p-5 border-b border-white/5">
                                    <h2 className="text-lg font-semibold text-white">Uploaded Videos</h2>
                                </div>
                                <div className="p-4 space-y-3">
                                    {videos.map((vid) => (
                                        <div key={vid.fileName} className="video-card">
                                            <div className="w-10 h-10 rounded-xl bg-primary-600/15 flex items-center justify-center text-lg flex-shrink-0">
                                                🎬
                                            </div>
                                            <div className="flex-1 min-w-0">
                                                <p className="text-dark-200 font-medium truncate">{vid.fileName}</p>
                                                <p className="text-dark-500 text-xs">
                                                    {formatSize(vid.fileSize)} • {formatDate(vid.uploadedAt)}
                                                </p>
                                            </div>
                                            <span className="badge badge-success flex-shrink-0">{vid.status}</span>
                                            <button
                                                onClick={(e) => { e.stopPropagation(); handleDeleteVideo(vid.fileName); }}
                                                className="btn-danger text-xs flex-shrink-0"
                                                style={{ padding: '6px 14px' }}
                                            >
                                                Delete
                                            </button>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}

                        {videos.length === 0 && !uploading && (
                            <div className="glass-card p-10 text-center">
                                <p className="text-dark-500">No videos uploaded yet. Upload one to get started!</p>
                            </div>
                        )}
                    </div>
                )}
            </div>

            {/* ── Camera Modal (unchanged) ── */}
            {showModal && (
                <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 animate-fade-in">
                    <div className="glass-card p-8 w-full max-w-md">
                        <h2 className="text-xl font-bold text-white mb-6">{editId ? 'Edit Camera' : 'Add New Camera'}</h2>
                        <form onSubmit={handleSubmit} className="space-y-4">
                            <div>
                                <label className="block text-sm font-medium text-dark-300 mb-2">Camera Name</label>
                                <input value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} required placeholder="e.g. Main Entrance" />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-dark-300 mb-2">RTSP URL</label>
                                <input value={form.rtspUrl} onChange={e => setForm({ ...form, rtspUrl: e.target.value })} required placeholder="rtsp://admin:pass@192.168.0.3:554/..." />
                            </div>
                            <div>
                                <label className="block text-sm font-medium text-dark-300 mb-2">Location</label>
                                <input value={form.location} onChange={e => setForm({ ...form, location: e.target.value })} placeholder="e.g. Building A, Floor 1" />
                            </div>
                            <div className="flex gap-3 pt-2">
                                <button type="submit" className="btn-primary flex-1">{editId ? 'Update' : 'Create'}</button>
                                <button type="button" onClick={() => setShowModal(false)} className="btn-secondary flex-1">Cancel</button>
                            </div>
                        </form>
                    </div>
                </div>
            )}
        </div>
    );
}
