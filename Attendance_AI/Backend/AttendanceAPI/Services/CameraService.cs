using Microsoft.EntityFrameworkCore;
using AttendanceAPI.Data;
using AttendanceAPI.DTOs;
using AttendanceAPI.Models;

namespace AttendanceAPI.Services;

public class CameraService
{
    private readonly AppDbContext _db;

    public CameraService(AppDbContext db) => _db = db;

    public async Task<List<CameraResponseDto>> GetAllAsync()
    {
        return await _db.Cameras
            .OrderBy(c => c.Name)
            .Select(c => new CameraResponseDto
            {
                Id = c.Id,
                Name = c.Name,
                RtspUrl = c.RtspUrl,
                Location = c.Location,
                IsActive = c.IsActive
            })
            .ToListAsync();
    }

    public async Task<CameraResponseDto?> GetByIdAsync(int id)
    {
        var cam = await _db.Cameras.FindAsync(id);
        if (cam == null) return null;
        return MapToDto(cam);
    }

    public async Task<CameraResponseDto> CreateAsync(CreateCameraDto dto)
    {
        var cam = new Camera
        {
            Name = dto.Name,
            RtspUrl = dto.RtspUrl,
            Location = dto.Location,
            IsActive = true
        };
        _db.Cameras.Add(cam);
        await _db.SaveChangesAsync();
        return MapToDto(cam);
    }

    public async Task<CameraResponseDto?> UpdateAsync(int id, UpdateCameraDto dto)
    {
        var cam = await _db.Cameras.FindAsync(id);
        if (cam == null) return null;

        if (dto.Name != null) cam.Name = dto.Name;
        if (dto.RtspUrl != null) cam.RtspUrl = dto.RtspUrl;
        if (dto.Location != null) cam.Location = dto.Location;
        if (dto.IsActive.HasValue) cam.IsActive = dto.IsActive.Value;

        await _db.SaveChangesAsync();
        return MapToDto(cam);
    }

    public async Task<bool> DeleteAsync(int id)
    {
        var cam = await _db.Cameras.FindAsync(id);
        if (cam == null) return false;
        _db.Cameras.Remove(cam);
        await _db.SaveChangesAsync();
        return true;
    }

    private static CameraResponseDto MapToDto(Camera c) => new()
    {
        Id = c.Id,
        Name = c.Name,
        RtspUrl = c.RtspUrl,
        Location = c.Location,
        IsActive = c.IsActive
    };
}
