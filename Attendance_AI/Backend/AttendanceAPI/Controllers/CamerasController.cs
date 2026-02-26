using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;
using AttendanceAPI.DTOs;
using AttendanceAPI.Services;

namespace AttendanceAPI.Controllers;

[ApiController]
[Route("api/[controller]")]
public class CamerasController : ControllerBase
{
    private readonly CameraService _service;
    private readonly IWebHostEnvironment _env;

    public CamerasController(CameraService service, IWebHostEnvironment env)
    {
        _service = service;
        _env = env;
    }

    // ── Camera CRUD (existing) ─────────────────────────────────

    [HttpGet]
    public async Task<ActionResult<List<CameraResponseDto>>> GetAll()
    {
        return Ok(await _service.GetAllAsync());
    }

    [HttpGet("{id}")]
    public async Task<ActionResult<CameraResponseDto>> GetById(int id)
    {
        var result = await _service.GetByIdAsync(id);
        return result == null ? NotFound() : Ok(result);
    }

    [HttpPost]
    [Authorize(Roles = "Admin")]
    public async Task<ActionResult<CameraResponseDto>> Create([FromBody] CreateCameraDto dto)
    {
        var result = await _service.CreateAsync(dto);
        return CreatedAtAction(nameof(GetById), new { id = result.Id }, result);
    }

    [HttpPut("{id}")]
    [Authorize(Roles = "Admin")]
    public async Task<ActionResult<CameraResponseDto>> Update(int id, [FromBody] UpdateCameraDto dto)
    {
        var result = await _service.UpdateAsync(id, dto);
        return result == null ? NotFound() : Ok(result);
    }

    [HttpDelete("{id}")]
    [Authorize(Roles = "Admin")]
    public async Task<ActionResult> Delete(int id)
    {
        return await _service.DeleteAsync(id) ? NoContent() : NotFound();
    }

    // ── Video Upload Endpoints ─────────────────────────────────

    private string VideosDir => Path.Combine(_env.WebRootPath ?? Path.Combine(Directory.GetCurrentDirectory(), "wwwroot"), "uploads", "videos");

    /// <summary>Upload a video file for AI processing.</summary>
    [HttpPost("upload-video")]
    [RequestSizeLimit(524_288_000)] // 500 MB
    public async Task<ActionResult<VideoUploadResponseDto>> UploadVideo(IFormFile file)
    {
        if (file == null || file.Length == 0)
            return BadRequest(new { message = "No file provided." });

        var allowed = new[] { ".mp4", ".avi", ".mkv", ".mov", ".wmv", ".webm" };
        var ext = Path.GetExtension(file.FileName).ToLowerInvariant();
        if (!allowed.Contains(ext))
            return BadRequest(new { message = $"Unsupported file type '{ext}'. Allowed: {string.Join(", ", allowed)}" });

        Directory.CreateDirectory(VideosDir);

        // unique file name to avoid collisions
        var safeName = $"{Path.GetFileNameWithoutExtension(file.FileName)}_{DateTime.UtcNow:yyyyMMddHHmmss}{ext}";
        var filePath = Path.Combine(VideosDir, safeName);

        using (var stream = new FileStream(filePath, FileMode.Create))
        {
            await file.CopyToAsync(stream);
        }

        return Ok(new VideoUploadResponseDto
        {
            FileName = safeName,
            FilePath = $"/uploads/videos/{safeName}",
            FileSize = file.Length,
            UploadedAt = DateTime.UtcNow,
            Status = "Uploaded"
        });
    }

    /// <summary>List all uploaded video files.</summary>
    [HttpGet("videos")]
    public ActionResult<List<VideoUploadResponseDto>> GetVideos()
    {
        if (!Directory.Exists(VideosDir))
            return Ok(new List<VideoUploadResponseDto>());

        var files = new DirectoryInfo(VideosDir)
            .GetFiles()
            .OrderByDescending(f => f.CreationTimeUtc)
            .Select(f => new VideoUploadResponseDto
            {
                FileName = f.Name,
                FilePath = $"/uploads/videos/{f.Name}",
                FileSize = f.Length,
                UploadedAt = f.CreationTimeUtc,
                Status = "Uploaded"
            })
            .ToList();

        return Ok(files);
    }

    /// <summary>Delete an uploaded video file.</summary>
    [HttpDelete("videos/{fileName}")]
    [Authorize(Roles = "Admin")]
    public ActionResult DeleteVideo(string fileName)
    {
        var filePath = Path.Combine(VideosDir, fileName);
        if (!System.IO.File.Exists(filePath))
            return NotFound(new { message = "Video file not found." });

        System.IO.File.Delete(filePath);
        return NoContent();
    }
}
