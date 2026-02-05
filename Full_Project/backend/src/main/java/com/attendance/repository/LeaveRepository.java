package com.attendance.repository;

import com.attendance.entity.Leave;
import com.attendance.entity.Leave.LeaveStatus;
import com.attendance.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface LeaveRepository extends JpaRepository<Leave, Long> {
    List<Leave> findByUser(User user);

    List<Leave> findByStatus(LeaveStatus status);

    List<Leave> findByUserOrderByCreatedAtDesc(User user);
}
