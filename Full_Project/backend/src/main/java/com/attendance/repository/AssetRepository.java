package com.attendance.repository;

import com.attendance.entity.Asset;
import com.attendance.entity.Asset.AssetStatus;
import com.attendance.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface AssetRepository extends JpaRepository<Asset, Long> {
    List<Asset> findByStatus(AssetStatus status);

    List<Asset> findByAssignedUser(User user);
}
