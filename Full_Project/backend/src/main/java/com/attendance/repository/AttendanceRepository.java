package com.attendance.repository;

import com.attendance.entity.Attendance;
import com.attendance.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.time.LocalDateTime;
import java.util.List;

@Repository
public interface AttendanceRepository extends JpaRepository<Attendance, Long> {
    List<Attendance> findByUser(User user);

    List<Attendance> findByUserOrderByTimestampDesc(User user);

    List<Attendance> findByTimestampBetween(LocalDateTime start, LocalDateTime end);

    List<Attendance> findByUserAndTimestampBetween(User user, LocalDateTime start, LocalDateTime end);
}
