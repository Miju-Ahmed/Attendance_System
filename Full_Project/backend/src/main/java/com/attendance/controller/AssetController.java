package com.attendance.controller;

import com.attendance.dto.AssetRequest;
import com.attendance.dto.MessageResponse;
import com.attendance.entity.Asset;
import com.attendance.entity.User;
import com.attendance.service.AssetService;
import com.attendance.service.AuthService;
import jakarta.validation.Valid;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.security.access.prepost.PreAuthorize;
import org.springframework.web.bind.annotation.*;

import java.util.List;

@RestController
@RequestMapping("/api/assets")
@RequiredArgsConstructor
public class AssetController {

    private final AssetService assetService;
    private final AuthService authService;

    @GetMapping
    public ResponseEntity<List<Asset>> getAllAssets() {
        return ResponseEntity.ok(assetService.getAllAssets());
    }

    @GetMapping("/{id}")
    public ResponseEntity<Asset> getAssetById(@PathVariable Long id) {
        return ResponseEntity.ok(assetService.getAssetById(id));
    }

    @PostMapping
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<Asset> createAsset(@Valid @RequestBody AssetRequest request) {
        return ResponseEntity.ok(assetService.createAsset(request));
    }

    @PostMapping("/{id}/request")
    public ResponseEntity<MessageResponse> requestAsset(@PathVariable Long id) {
        User currentUser = authService.getCurrentUser();
        assetService.requestAsset(id, currentUser);
        return ResponseEntity.ok(new MessageResponse("Asset request submitted successfully"));
    }

    @PutMapping("/{id}/assign")
    @PreAuthorize("hasRole('ADMIN')")
    public ResponseEntity<Asset> assignAsset(@PathVariable Long id, @RequestParam Long userId) {
        return ResponseEntity.ok(assetService.assignAsset(id, userId));
    }

    @GetMapping("/my-assets")
    public ResponseEntity<List<Asset>> getMyAssets() {
        User currentUser = authService.getCurrentUser();
        return ResponseEntity.ok(assetService.getUserAssets(currentUser));
    }
}
