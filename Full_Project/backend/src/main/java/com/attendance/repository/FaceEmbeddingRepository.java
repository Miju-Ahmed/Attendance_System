package com.attendance.repository;

import com.attendance.entity.FaceEmbedding;
import com.attendance.entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface FaceEmbeddingRepository extends JpaRepository<FaceEmbedding, Long> {
    List<FaceEmbedding> findByUser(User user);
}
