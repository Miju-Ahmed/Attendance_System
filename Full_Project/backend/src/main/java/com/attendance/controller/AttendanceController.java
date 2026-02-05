package com.attendance.controller;

import com.attendance.dto.MessageResponse;
import com.attendance.entity.Attendance;
import com.attendance.entity.User;
import com.attendance.service.AttendanceService;
import com.attendance.service.AuthService;
import lombok.RequiredArgsConstructor;
import org.springframework.format.annotation.DateTimeFormat;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.List;

@RestController
@RequestMapping("/api/attendance")
@RequiredArgsConstructor
public class AttendanceController {

    private final AttendanceService attendanceService;
    private final AuthService authService;
    private final com.attendance.service.UserService userService;

    @GetMapping
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<List<Attendance>> getAllAttendance() {
        return ResponseEntity.ok(attendanceService.getAllAttendance());
    }

    @GetMapping("/user/{userId}")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<List<Attendance>> getAttendanceByUserId(@PathVariable Long userId) {
        User user = userService.getUserById(userId);
        return ResponseEntity.ok(attendanceService.getUserAttendance(user));
    }

    @GetMapping("/my-attendance")
    public ResponseEntity<List<Attendance>> getMyAttendance() {
        User currentUser = authService.getCurrentUser();
        return ResponseEntity.ok(attendanceService.getUserAttendance(currentUser));
    }

    @GetMapping("/range")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<List<Attendance>> getAttendanceByRange(
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime start,
            @RequestParam @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) LocalDateTime end) {
        return ResponseEntity.ok(attendanceService.getAttendanceByDateRange(start, end));
    }

    @PostMapping("/sync")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<MessageResponse> syncFromSQLite() {
        int count = attendanceService.syncFromSQLite();
        return ResponseEntity.ok(new MessageResponse("Synced " + count + " attendance records"));
    }
}
