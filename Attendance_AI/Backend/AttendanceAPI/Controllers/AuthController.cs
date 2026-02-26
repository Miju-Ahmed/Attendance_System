using Microsoft.AspNetCore.Mvc;
using AttendanceAPI.DTOs;
using AttendanceAPI.Services;

namespace AttendanceAPI.Controllers;

[ApiController]
[Route("api/[controller]")]
public class AuthController : ControllerBase
{
    private readonly AuthService _service;

    public AuthController(AuthService service) => _service = service;

    [HttpPost("login")]
    public async Task<ActionResult<TokenResponseDto>> Login([FromBody] LoginDto dto)
    {
        var result = await _service.LoginAsync(dto);
        if (result == null)
            return Unauthorized(new { message = "Invalid username or password." });
        return Ok(result);
    }

    [HttpPost("register")]
    public async Task<ActionResult<TokenResponseDto>> Register([FromBody] RegisterDto dto)
    {
        var result = await _service.RegisterAsync(dto);
        if (result == null)
            return Conflict(new { message = "Username already exists." });
        return Ok(result);
    }
}
