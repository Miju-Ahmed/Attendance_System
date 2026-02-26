using System.ComponentModel.DataAnnotations;

namespace AttendanceAPI.Models;

public class Camera
{
    [Key]
    public int Id { get; set; }

    [Required, MaxLength(100)]
    public string Name { get; set; } = string.Empty;

    [Required, MaxLength(500)]
    public string RtspUrl { get; set; } = string.Empty;

    [MaxLength(200)]
    public string Location { get; set; } = string.Empty;

    public bool IsActive { get; set; } = true;
}
