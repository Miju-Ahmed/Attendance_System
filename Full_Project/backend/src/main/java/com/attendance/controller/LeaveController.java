package com.attendance.controller;

import com.attendance.dto.LeaveRequest;
import com.attendance.dto.MessageResponse;
import com.attendance.entity.Leave;
import com.attendance.entity.User;
import com.attendance.service.AuthService;
import com.attendance.service.LeaveService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/leaves")
@RequiredArgsConstructor
public class LeaveController {

    private final LeaveService leaveService;
    private final AuthService authService;

    @GetMapping
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<List<Leave>> getAllLeaves() {
        return ResponseEntity.ok(leaveService.getAllLeaves());
    }

    @GetMapping("/pending")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<List<Leave>> getPendingLeaves() {
        return ResponseEntity.ok(leaveService.getPendingLeaves());
    }

    @PostMapping
    public ResponseEntity<MessageResponse> applyLeave(@Valid @RequestBody LeaveRequest request) {
        User currentUser = authService.getCurrentUser();
        leaveService.applyLeave(request, currentUser);
        return ResponseEntity.ok(new MessageResponse("Leave application submitted successfully"));
    }

    @PutMapping("/{id}/approve")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<Leave> approveLeave(@PathVariable Long id) {
        return ResponseEntity.ok(leaveService.approveLeave(id));
    }

    @PutMapping("/{id}/reject")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<Leave> rejectLeave(@PathVariable Long id) {
        return ResponseEntity.ok(leaveService.rejectLeave(id));
    }

    @GetMapping("/my-leaves")
    public ResponseEntity<List<Leave>> getMyLeaves() {
        User currentUser = authService.getCurrentUser();
        return ResponseEntity.ok(leaveService.getUserLeaves(currentUser));
    }
}
