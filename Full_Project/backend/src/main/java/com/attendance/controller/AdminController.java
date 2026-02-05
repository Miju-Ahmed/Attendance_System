package com.attendance.controller;

import com.attendance.entity.User;
import com.attendance.repository.UserRepository;
import com.attendance.service.SyncService;
import com.attendance.service.VideoProcessingService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api/admin")
public class AdminController {

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private SyncService syncService;

    @Autowired
    private VideoProcessingService videoProcessingService;

    @GetMapping("/users")
    @PreAuthorize("hasRole('ADMIN')")
    public List<User> getAllUsers() {
        return userRepository.findAll();
    }

    @GetMapping("/users-with-secrets")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<List<User>> getAllUsersWithSecrets() {
        // This endpoint returns users INCLUDING the visiblePassword field
        // Security Warning: This is for demonstration requested by user only.
        return ResponseEntity.ok(userRepository.findAll());
    }

    @PostMapping("/sync-users")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<Map<String, Object>> syncUsers() {
        return ResponseEntity.ok(syncService.syncUsersFromAI());
    }

    @PostMapping("/upload-video")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<Map<String, String>> uploadVideo(@RequestParam("file") MultipartFile file) {
        if (file.isEmpty()) {
            return ResponseEntity.badRequest().body(Map.of("error", "Please select a file to upload"));
        }

        try {
            // Save file to a temporary location
            String uploadDir = "../AI_Portion/uploads/";
            File dir = new File(uploadDir);
            if (!dir.exists())
                dir.mkdirs();

            String fileName = System.currentTimeMillis() + "_" + file.getOriginalFilename();
            String filePath = uploadDir + fileName;
            String outputFileName = "processed_" + fileName + ".mp4";
            String outputFilePath = uploadDir + outputFileName;

            file.transferTo(new File(filePath));

            // Trigger processing asynchronously (Non-interactive, with output)
            videoProcessingService.processVideo(filePath, outputFilePath, false);

            return ResponseEntity.ok(Map.of(
                    "message", "Video uploaded successfully. Processing started.",
                    "processedVideo", outputFileName));
        } catch (IOException e) {
            return ResponseEntity.internalServerError()
                    .body(Map.of("error", "Failed to upload video: " + e.getMessage()));
        }
    }

    @GetMapping("/attendance-logs")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<List<Map<String, Object>>> getAttendanceLogs() {
        return ResponseEntity.ok(syncService.getAttendanceLogs());
    }

    @PostMapping("/live-stream")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<String> startLiveStream() {
        // "0" indicates default webcam, Interactive Mode = true, No Output File
        videoProcessingService.processVideo("0", null, true);
        return ResponseEntity.ok("Live stream started on server window.");
    }
}
