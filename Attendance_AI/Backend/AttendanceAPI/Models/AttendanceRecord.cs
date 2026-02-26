using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace AttendanceAPI.Models;

public class AttendanceRecord
{
    [Key]
    public int Id { get; set; }

    public int? EmployeeId { get; set; }

    [Column(TypeName = "date")]
    public DateTime Date { get; set; }

    public DateTime? EntryTime { get; set; }

    public DateTime? ExitTime { get; set; }

    public double Confidence { get; set; }

    [MaxLength(50)]
    public string? CameraId { get; set; }

    [MaxLength(100)]
    public string? EmployeeName { get; set; }

    public bool IsUnknown { get; set; } = false;

    // Navigation
    [ForeignKey("EmployeeId")]
    public Employee? Employee { get; set; }
}
