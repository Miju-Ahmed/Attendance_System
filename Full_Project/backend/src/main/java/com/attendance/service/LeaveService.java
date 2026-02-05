package com.attendance.service;

import com.attendance.dto.LeaveRequest;
import com.attendance.entity.Leave;
import com.attendance.entity.User;
import com.attendance.exception.ResourceNotFoundException;
import com.attendance.repository.LeaveRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@RequiredArgsConstructor
public class LeaveService {

    private final LeaveRepository leaveRepository;

    public List<Leave> getAllLeaves() {
        return leaveRepository.findAll();
    }

    public Leave getLeaveById(Long id) {
        return leaveRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Leave not found with id: " + id));
    }

    @Transactional
    public Leave applyLeave(LeaveRequest request, User user) {
        Leave leave = new Leave();
        leave.setUser(user);
        leave.setLeaveType(request.getLeaveType());
        leave.setStartDate(request.getStartDate());
        leave.setEndDate(request.getEndDate());
        leave.setReason(request.getReason());
        leave.setStatus(Leave.LeaveStatus.PENDING);
        return leaveRepository.save(leave);
    }

    @Transactional
    public Leave approveLeave(Long leaveId) {
        Leave leave = getLeaveById(leaveId);
        leave.setStatus(Leave.LeaveStatus.APPROVED);
        return leaveRepository.save(leave);
    }

    @Transactional
    public Leave rejectLeave(Long leaveId) {
        Leave leave = getLeaveById(leaveId);
        leave.setStatus(Leave.LeaveStatus.REJECTED);
        return leaveRepository.save(leave);
    }

    public List<Leave> getUserLeaves(User user) {
        return leaveRepository.findByUserOrderByCreatedAtDesc(user);
    }

    public List<Leave> getPendingLeaves() {
        return leaveRepository.findByStatus(Leave.LeaveStatus.PENDING);
    }
}
