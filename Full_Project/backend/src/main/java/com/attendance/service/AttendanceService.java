package com.attendance.service;

import com.attendance.entity.Attendance;
import com.attendance.entity.User;
import com.attendance.repository.AttendanceRepository;
import com.attendance.repository.UserRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.sql.*;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.List;

@Service
@RequiredArgsConstructor
@Slf4j
public class AttendanceService {

    private final AttendanceRepository attendanceRepository;
    private final UserRepository userRepository;

    @Value("${sqlite.db.path}")
    private String sqliteDbPath;

    public List<Attendance> getAllAttendance() {
        return attendanceRepository.findAll();
    }

    public List<Attendance> getUserAttendance(User user) {
        return attendanceRepository.findByUserOrderByTimestampDesc(user);
    }

    public List<Attendance> getAttendanceByDateRange(LocalDateTime start, LocalDateTime end) {
        return attendanceRepository.findByTimestampBetween(start, end);
    }

    @Transactional
    public int syncFromSQLite() {
        int syncedCount = 0;

        try (Connection conn = DriverManager.getConnection("jdbc:sqlite:" + sqliteDbPath)) {
            String query = "SELECT person_name, stable_id, event_type, timestamp, confidence FROM attendance ORDER BY timestamp DESC";
            Statement stmt = conn.createStatement();
            ResultSet rs = stmt.executeQuery(query);

            DateTimeFormatter formatter = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

            while (rs.next()) {
                int stableId = rs.getInt("stable_id");
                String eventType = rs.getString("event_type");
                String timestampStr = rs.getString("timestamp");
                double confidence = rs.getDouble("confidence");

                LocalDateTime timestamp = LocalDateTime.parse(timestampStr, formatter);

                // Find user by stable_id
                User user = userRepository.findByStableId(stableId).orElse(null);

                if (user != null) {
                    // Check if this record already exists
                    List<Attendance> existing = attendanceRepository.findByUserAndTimestampBetween(
                            user,
                            timestamp.minusSeconds(1),
                            timestamp.plusSeconds(1));

                    if (existing.isEmpty()) {
                        Attendance attendance = new Attendance();
                        attendance.setUser(user);
                        attendance.setStableId(stableId);
                        attendance.setEventType(eventType);
                        attendance.setTimestamp(timestamp);
                        attendance.setConfidence(confidence);
                        attendanceRepository.save(attendance);
                        syncedCount++;
                    }
                } else {
                    log.warn("No user found for stable_id: {}", stableId);
                }
            }

            log.info("Synced {} attendance records from SQLite", syncedCount);
        } catch (SQLException e) {
            log.error("Error syncing from SQLite: {}", e.getMessage());
            throw new RuntimeException("Failed to sync attendance data: " + e.getMessage());
        }

        return syncedCount;
    }
}
