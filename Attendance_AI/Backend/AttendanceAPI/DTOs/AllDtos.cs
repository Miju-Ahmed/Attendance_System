using System.ComponentModel.DataAnnotations;

namespace AttendanceAPI.DTOs;

// --- Attendance DTOs ---

public class MarkAttendanceDto
{
    public int? EmployeeId { get; set; }
    public string? EmployeeName { get; set; }
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;
    public double Confidence { get; set; }
    public string? CameraId { get; set; }
    public bool IsUnknown { get; set; } = false;
}

public class AttendanceResponseDto
{
    public int Id { get; set; }
    public int? EmployeeId { get; set; }
    public string? EmployeeName { get; set; }
    public DateTime Date { get; set; }
    public DateTime? EntryTime { get; set; }
    public DateTime? ExitTime { get; set; }
    public double Confidence { get; set; }
    public string? CameraId { get; set; }
    public bool IsUnknown { get; set; }
}

public class AttendanceReportDto
{
    public DateTime StartDate { get; set; }
    public DateTime EndDate { get; set; }
    public int? EmployeeId { get; set; }
    public string? Department { get; set; }
}

public class DashboardStatsDto
{
    public int TotalEmployees { get; set; }
    public int PresentToday { get; set; }
    public int AbsentToday { get; set; }
    public int LateEmployees { get; set; }
    public List<AttendanceResponseDto> RecentLogs { get; set; } = new();
}

// --- Employee DTOs ---

public class CreateEmployeeDto
{
    [Required, MaxLength(100)]
    public string Name { get; set; } = string.Empty;

    [MaxLength(100)]
    public string Department { get; set; } = string.Empty;

    [MaxLength(100)]
    public string Position { get; set; } = string.Empty;

    public string? FaceEmbeddingPath { get; set; }
}

public class UpdateEmployeeDto
{
    [MaxLength(100)]
    public string? Name { get; set; }

    [MaxLength(100)]
    public string? Department { get; set; }

    [MaxLength(100)]
    public string? Position { get; set; }

    public string? FaceEmbeddingPath { get; set; }
    public bool? IsActive { get; set; }
}

public class EmployeeResponseDto
{
    public int Id { get; set; }
    public string Name { get; set; } = string.Empty;
    public string Department { get; set; } = string.Empty;
    public string Position { get; set; } = string.Empty;
    public string? FaceEmbeddingPath { get; set; }
    public DateTime CreatedAt { get; set; }
    public bool IsActive { get; set; }
}

// --- Camera DTOs ---

public class CreateCameraDto
{
    [Required, MaxLength(100)]
    public string Name { get; set; } = string.Empty;

    [Required, MaxLength(500)]
    public string RtspUrl { get; set; } = string.Empty;

    [MaxLength(200)]
    public string Location { get; set; } = string.Empty;
}

public class UpdateCameraDto
{
    public string? Name { get; set; }
    public string? RtspUrl { get; set; }
    public string? Location { get; set; }
    public bool? IsActive { get; set; }
}

public class CameraResponseDto
{
    public int Id { get; set; }
    public string Name { get; set; } = string.Empty;
    public string RtspUrl { get; set; } = string.Empty;
    public string Location { get; set; } = string.Empty;
    public bool IsActive { get; set; }
}

// --- Video Upload DTOs ---

public class VideoUploadResponseDto
{
    public string FileName { get; set; } = string.Empty;
    public string FilePath { get; set; } = string.Empty;
    public long FileSize { get; set; }
    public DateTime UploadedAt { get; set; }
    public string Status { get; set; } = "Uploaded";
}

// --- Auth DTOs ---

public class LoginDto
{
    [Required]
    public string Username { get; set; } = string.Empty;

    [Required]
    public string Password { get; set; } = string.Empty;
}

public class RegisterDto
{
    [Required, MaxLength(100)]
    public string Username { get; set; } = string.Empty;

    [Required, MinLength(6)]
    public string Password { get; set; } = string.Empty;

    [MaxLength(50)]
    public string Role { get; set; } = "User";
}

public class TokenResponseDto
{
    public string Token { get; set; } = string.Empty;
    public string Username { get; set; } = string.Empty;
    public string Role { get; set; } = string.Empty;
    public DateTime ExpiresAt { get; set; }
}
