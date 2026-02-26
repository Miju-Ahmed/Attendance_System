using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using AttendanceAPI.DTOs;
using AttendanceAPI.Services;

namespace AttendanceAPI.Controllers;

[ApiController]
[Route("api/[controller]")]
public class AttendanceController : ControllerBase
{
    private readonly AttendanceService _service;

    public AttendanceController(AttendanceService service) => _service = service;

    /// <summary>
    /// Mark attendance (called by Python AI script or manually).
    /// No auth required — AI script calls this endpoint directly.
    /// </summary>
    [HttpPost("mark")]
    public async Task<ActionResult<AttendanceResponseDto>> MarkAttendance([FromBody] MarkAttendanceDto dto)
    {
        var result = await _service.MarkAttendanceAsync(dto);
        return Ok(result);
    }

    /// <summary>Get today's attendance records.</summary>
    [HttpGet("today")]
    public async Task<ActionResult<List<AttendanceResponseDto>>> GetTodayAttendance()
    {
        return Ok(await _service.GetTodayAttendanceAsync());
    }

    /// <summary>Get attendance report for a date range.</summary>
    [HttpGet("report")]
    public async Task<ActionResult<List<AttendanceResponseDto>>> GetReport(
        [FromQuery] DateTime startDate, [FromQuery] DateTime endDate,
        [FromQuery] int? employeeId, [FromQuery] string? department)
    {
        var dto = new AttendanceReportDto
        {
            StartDate = startDate,
            EndDate = endDate,
            EmployeeId = employeeId,
            Department = department
        };
        return Ok(await _service.GetAttendanceReportAsync(dto));
    }

    /// <summary>Get dashboard statistics.</summary>
    [HttpGet("dashboard")]
    public async Task<ActionResult<DashboardStatsDto>> GetDashboardStats()
    {
        return Ok(await _service.GetDashboardStatsAsync());
    }
}
