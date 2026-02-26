using Microsoft.EntityFrameworkCore;
using AttendanceAPI.Data;
using AttendanceAPI.DTOs;
using AttendanceAPI.Models;

namespace AttendanceAPI.Services;

public class EmployeeService
{
    private readonly AppDbContext _db;

    public EmployeeService(AppDbContext db) => _db = db;

    public async Task<List<EmployeeResponseDto>> GetAllAsync()
    {
        return await _db.Employees
            .OrderBy(e => e.Name)
            .Select(e => MapToDto(e))
            .ToListAsync();
    }

    public async Task<EmployeeResponseDto?> GetByIdAsync(int id)
    {
        var employee = await _db.Employees.FindAsync(id);
        return employee == null ? null : MapToDto(employee);
    }

    public async Task<EmployeeResponseDto> CreateAsync(CreateEmployeeDto dto)
    {
        var employee = new Employee
        {
            Name = dto.Name,
            Department = dto.Department,
            Position = dto.Position,
            FaceEmbeddingPath = dto.FaceEmbeddingPath,
            CreatedAt = DateTime.UtcNow,
            IsActive = true
        };
        _db.Employees.Add(employee);
        await _db.SaveChangesAsync();
        return MapToDto(employee);
    }

    public async Task<EmployeeResponseDto?> UpdateAsync(int id, UpdateEmployeeDto dto)
    {
        var employee = await _db.Employees.FindAsync(id);
        if (employee == null) return null;

        if (dto.Name != null) employee.Name = dto.Name;
        if (dto.Department != null) employee.Department = dto.Department;
        if (dto.Position != null) employee.Position = dto.Position;
        if (dto.FaceEmbeddingPath != null) employee.FaceEmbeddingPath = dto.FaceEmbeddingPath;
        if (dto.IsActive.HasValue) employee.IsActive = dto.IsActive.Value;

        await _db.SaveChangesAsync();
        return MapToDto(employee);
    }

    public async Task<bool> DeleteAsync(int id)
    {
        var employee = await _db.Employees.FindAsync(id);
        if (employee == null) return false;
        _db.Employees.Remove(employee);
        await _db.SaveChangesAsync();
        return true;
    }

    private static EmployeeResponseDto MapToDto(Employee e) => new()
    {
        Id = e.Id,
        Name = e.Name,
        Department = e.Department,
        Position = e.Position,
        FaceEmbeddingPath = e.FaceEmbeddingPath,
        CreatedAt = e.CreatedAt,
        IsActive = e.IsActive
    };
}
