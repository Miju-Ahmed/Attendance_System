package com.attendance.controller;

import com.attendance.dto.MessageResponse;
import com.attendance.entity.User;
import com.attendance.service.AuthService;
import com.attendance.service.UserService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.util.List;

@RestController
@RequestMapping("/api/users")
@RequiredArgsConstructor
public class UserController {

    private final UserService userService;
    private final AuthService authService;

    @GetMapping
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<List<User>> getAllUsers() {
        return ResponseEntity.ok(userService.getAllUsers());
    }

    @GetMapping("/{id}")
    public ResponseEntity<User> getUserById(@PathVariable Long id) {
        return ResponseEntity.ok(userService.getUserById(id));
    }

    @PutMapping("/{id}")
    public ResponseEntity<User> updateUser(@PathVariable Long id, @RequestBody User userDetails) {
        User currentUser = authService.getCurrentUser();

        // Users can only update their own profile, admins can update anyone
        if (!currentUser.getRole().equals(User.Role.ADMIN) && !currentUser.getId().equals(id)) {
            return ResponseEntity.status(HttpStatus.FORBIDDEN).build();
        }

        return ResponseEntity.ok(userService.updateUser(id, userDetails));
    }

    @PostMapping("/{id}/photo")
    public ResponseEntity<MessageResponse> uploadProfilePhoto(
            @PathVariable Long id,
            @RequestParam("file") MultipartFile file) throws IOException {

        User currentUser = authService.getCurrentUser();

        // Users can only upload their own photo, admins can upload for anyone
        if (!currentUser.getRole().equals(User.Role.ADMIN) && !currentUser.getId().equals(id)) {
            return ResponseEntity.status(HttpStatus.FORBIDDEN).build();
        }

        userService.uploadProfilePhoto(id, file);
        return ResponseEntity.ok(new MessageResponse("Profile photo uploaded successfully"));
    }

    @GetMapping("/photo/{filename}")
    public ResponseEntity<byte[]> getProfilePhoto(@PathVariable String filename) throws IOException {
        byte[] image = userService.getProfilePhoto(filename);
        return ResponseEntity.ok()
                .contentType(MediaType.IMAGE_JPEG)
                .body(image);
    }
}
