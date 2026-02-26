using Microsoft.EntityFrameworkCore;
using AttendanceAPI.Data;
using AttendanceAPI.DTOs;
using AttendanceAPI.Models;

namespace AttendanceAPI.Services;

public class AttendanceService
{
    private readonly AppDbContext _db;
    private readonly IConfiguration _config;

    public AttendanceService(AppDbContext db, IConfiguration config)
    {
        _db = db;
        _config = config;
    }

    /// <summary>
    /// Core attendance marking logic:
    /// - If no record today → create with EntryTime
    /// - If has EntryTime but no ExitTime → set ExitTime
    /// - Configurable cooldown prevents duplicate entries
    /// </summary>
    public async Task<AttendanceResponseDto> MarkAttendanceAsync(MarkAttendanceDto dto)
    {
        var cooldownMinutes = _config.GetValue("AttendanceSettings:CooldownMinutes", 5);
        var now = dto.Timestamp;
        var today = now.Date;

        // Handle unknown detections
        if (dto.IsUnknown || dto.EmployeeId == null)
        {
            var unknownRecord = new AttendanceRecord
            {
                EmployeeId = null,
                EmployeeName = dto.EmployeeName ?? "Unknown",
                Date = today,
                EntryTime = now,
                Confidence = dto.Confidence,
                CameraId = dto.CameraId,
                IsUnknown = true
            };
            _db.AttendanceRecords.Add(unknownRecord);
            await _db.SaveChangesAsync();
            return MapToDto(unknownRecord);
        }

        // Get employee name
        var employee = await _db.Employees.FindAsync(dto.EmployeeId);
        var empName = employee?.Name ?? dto.EmployeeName ?? "Unknown";

        // Find today's record for this employee
        var existingRecord = await _db.AttendanceRecords
            .Where(a => a.EmployeeId == dto.EmployeeId && a.Date == today && !a.IsUnknown)
            .OrderByDescending(a => a.Id)
            .FirstOrDefaultAsync();

        // Cooldown check — prevent duplicate entries within N minutes
        if (existingRecord != null)
        {
            var lastTime = existingRecord.ExitTime ?? existingRecord.EntryTime;
            if (lastTime.HasValue && (now - lastTime.Value).TotalMinutes < cooldownMinutes)
            {
                return MapToDto(existingRecord); // Skip, within cooldown
            }
        }

        if (existingRecord == null)
        {
            // No record today → mark ENTRY
            var newRecord = new AttendanceRecord
            {
                EmployeeId = dto.EmployeeId,
                EmployeeName = empName,
                Date = today,
                EntryTime = now,
                Confidence = dto.Confidence,
                CameraId = dto.CameraId,
                IsUnknown = false
            };
            _db.AttendanceRecords.Add(newRecord);
            await _db.SaveChangesAsync();
            return MapToDto(newRecord);
        }
        else if (existingRecord.ExitTime == null)
        {
            // Has EntryTime but no ExitTime → mark EXIT
            existingRecord.ExitTime = now;
            existingRecord.Confidence = Math.Max(existingRecord.Confidence, dto.Confidence);
            await _db.SaveChangesAsync();
            return MapToDto(existingRecord);
        }
        else
        {
            // Already has both entry and exit — create a new entry record
            var newRecord = new AttendanceRecord
            {
                EmployeeId = dto.EmployeeId,
                EmployeeName = empName,
                Date = today,
                EntryTime = now,
                Confidence = dto.Confidence,
                CameraId = dto.CameraId,
                IsUnknown = false
            };
            _db.AttendanceRecords.Add(newRecord);
            await _db.SaveChangesAsync();
            return MapToDto(newRecord);
        }
    }

    public async Task<List<AttendanceResponseDto>> GetTodayAttendanceAsync()
    {
        var today = DateTime.UtcNow.Date;
        var records = await _db.AttendanceRecords
            .Include(a => a.Employee)
            .Where(a => a.Date == today)
            .OrderByDescending(a => a.EntryTime)
            .ToListAsync();

        return records.Select(MapToDto).ToList();
    }

    public async Task<List<AttendanceResponseDto>> GetAttendanceReportAsync(AttendanceReportDto dto)
    {
        var query = _db.AttendanceRecords
            .Include(a => a.Employee)
            .Where(a => a.Date >= dto.StartDate.Date && a.Date <= dto.EndDate.Date);

        if (dto.EmployeeId.HasValue)
            query = query.Where(a => a.EmployeeId == dto.EmployeeId);

        if (!string.IsNullOrEmpty(dto.Department))
            query = query.Where(a => a.Employee != null && a.Employee.Department == dto.Department);

        var records = await query.OrderByDescending(a => a.Date).ThenByDescending(a => a.EntryTime).ToListAsync();
        return records.Select(MapToDto).ToList();
    }

    public async Task<DashboardStatsDto> GetDashboardStatsAsync()
    {
        var today = DateTime.UtcNow.Date;
        var totalEmployees = await _db.Employees.Where(e => e.IsActive).CountAsync();
        var presentToday = await _db.AttendanceRecords
            .Where(a => a.Date == today && !a.IsUnknown && a.EmployeeId != null)
            .Select(a => a.EmployeeId)
            .Distinct()
            .CountAsync();

        var lateThreshold = _config.GetValue("AttendanceSettings:LateThresholdHour", 9);
        var lateEmployees = await _db.AttendanceRecords
            .Where(a => a.Date == today && !a.IsUnknown && a.EntryTime.HasValue && a.EntryTime.Value.Hour >= lateThreshold)
            .Select(a => a.EmployeeId)
            .Distinct()
            .CountAsync();

        var recentLogs = await _db.AttendanceRecords
            .Include(a => a.Employee)
            .Where(a => a.Date == today)
            .OrderByDescending(a => a.EntryTime)
            .Take(20)
            .ToListAsync();

        return new DashboardStatsDto
        {
            TotalEmployees = totalEmployees,
            PresentToday = presentToday,
            AbsentToday = totalEmployees - presentToday,
            LateEmployees = lateEmployees,
            RecentLogs = recentLogs.Select(MapToDto).ToList()
        };
    }

    private static AttendanceResponseDto MapToDto(AttendanceRecord r) => new()
    {
        Id = r.Id,
        EmployeeId = r.EmployeeId,
        EmployeeName = r.EmployeeName ?? r.Employee?.Name ?? "Unknown",
        Date = r.Date,
        EntryTime = r.EntryTime,
        ExitTime = r.ExitTime,
        Confidence = r.Confidence,
        CameraId = r.CameraId,
        IsUnknown = r.IsUnknown
    };
}
