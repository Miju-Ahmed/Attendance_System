package com.attendance.service;

import com.attendance.dto.AssetRequest;
import com.attendance.entity.Asset;
import com.attendance.entity.User;
import com.attendance.exception.ResourceNotFoundException;
import com.attendance.repository.AssetRepository;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import java.util.List;

@Service
@RequiredArgsConstructor
public class AssetService {

    private final AssetRepository assetRepository;

    public List<Asset> getAllAssets() {
        return assetRepository.findAll();
    }

    public Asset getAssetById(Long id) {
        return assetRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Asset not found with id: " + id));
    }

    @Transactional
    public Asset createAsset(AssetRequest request) {
        Asset asset = new Asset();
        asset.setAssetName(request.getAssetName());
        asset.setAssetType(request.getAssetType());
        asset.setStatus(Asset.AssetStatus.AVAILABLE);
        return assetRepository.save(asset);
    }

    @Transactional
    public Asset requestAsset(Long assetId, User user) {
        Asset asset = getAssetById(assetId);

        if (asset.getStatus() != Asset.AssetStatus.AVAILABLE) {
            throw new RuntimeException("Asset is not available");
        }

        asset.setStatus(Asset.AssetStatus.REQUESTED);
        asset.setAssignedUser(user);
        return assetRepository.save(asset);
    }

    @Transactional
    public Asset assignAsset(Long assetId, Long userId) {
        Asset asset = getAssetById(assetId);
        asset.setStatus(Asset.AssetStatus.ASSIGNED);
        return assetRepository.save(asset);
    }

    public List<Asset> getUserAssets(User user) {
        return assetRepository.findByAssignedUser(user);
    }
}
