using Microsoft.EntityFrameworkCore;
using AttendanceAPI.Models;

namespace AttendanceAPI.Data;

public class AppDbContext : DbContext
{
    public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

    public DbSet<Employee> Employees => Set<Employee>();
    public DbSet<AttendanceRecord> AttendanceRecords => Set<AttendanceRecord>();
    public DbSet<Camera> Cameras => Set<Camera>();
    public DbSet<User> Users => Set<User>();

    protected override void OnModelCreating(ModelBuilder modelBuilder)
    {
        base.OnModelCreating(modelBuilder);

        modelBuilder.Entity<Employee>(entity =>
        {
            entity.HasIndex(e => e.Name);
            entity.HasIndex(e => e.IsActive);
        });

        modelBuilder.Entity<AttendanceRecord>(entity =>
        {
            entity.HasIndex(e => e.Date);
            entity.HasIndex(e => new { e.EmployeeId, e.Date });
            entity.HasOne(a => a.Employee)
                  .WithMany(e => e.AttendanceRecords)
                  .HasForeignKey(a => a.EmployeeId)
                  .OnDelete(DeleteBehavior.SetNull);
        });

        modelBuilder.Entity<User>(entity =>
        {
            entity.HasIndex(e => e.Username).IsUnique();
        });

        // Seed default admin user (password: Admin@123)
        modelBuilder.Entity<User>().HasData(new User
        {
            Id = 1,
            Username = "admin",
            PasswordHash = BCrypt.Net.BCrypt.HashPassword("Admin@123"),
            Role = "Admin"
        });
    }
}
