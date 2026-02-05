package com.attendance.dto;

import jakarta.validation.constraints.NotBlank;
import lombok.Data;

@Data
public class AssetRequest {

    @NotBlank(message = "Asset name is required")
    private String assetName;

    @NotBlank(message = "Asset type is required")
    private String assetType;
}
