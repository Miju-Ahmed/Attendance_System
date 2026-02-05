package com.attendance.service;

import com.attendance.entity.User;
import com.attendance.repository.UserRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;

import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;

import javax.sql.DataSource;
import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
public class SyncService {

    private static final Logger log = LoggerFactory.getLogger(SyncService.class);

    @Autowired
    private UserRepository userRepository;

    @Autowired
    private PasswordEncoder passwordEncoder;

    @Autowired
    private DataSource dataSource;

    // Removed @Transactional to allow partial success
    public Map<String, Object> syncUsersFromAI() {
        Map<String, Object> result = new HashMap<>();
        int added = 0;
        int skipped = 0;
        int errors = 0;
        List<String> logs = new ArrayList<>();

        log.info("Starting sync of users from face_embeddings table...");
        logs.add("Starting sync process...");

        List<Map<String, Object>> usersToCreate = new ArrayList<>();

        // Step 1: Read all users from DB (Raw JDBC)
        try (Connection conn = dataSource.getConnection()) {
            String sql = "SELECT DISTINCT person_name, stable_id FROM face_embeddings WHERE person_name IS NOT NULL AND stable_id IS NOT NULL";
            try (PreparedStatement stmt = conn.prepareStatement(sql)) {
                try (ResultSet rs = stmt.executeQuery()) {
                    while (rs.next()) {
                        Map<String, Object> userData = new HashMap<>();
                        userData.put("name", rs.getString("person_name"));
                        userData.put("stableId", rs.getInt("stable_id"));
                        usersToCreate.add(userData);
                    }
                }
            }
        } catch (Exception e) {
            log.error("Database connection error during sync read: {}", e.getMessage());
            logs.add("Critical error reading DB: " + e.getMessage());
            result.put("status", "error");
            result.put("message", e.getMessage());
            return result;
        }

        // Step 2: Write users to DB (JPA) - Connection is closed now, so no lock
        for (Map<String, Object> userData : usersToCreate) {
            String name = (String) userData.get("name");
            Integer stableId = (Integer) userData.get("stableId");

            try {
                // Determine email based on name
                String email = name.toLowerCase().replaceAll("\\s+", ".") + "@employee.com";

                if (userRepository.existsByEmail(email)) {
                    log.info("User {} already exists (email: {})", name, email);
                    skipped++;
                    continue;
                }

                // Create new user
                User user = new User();
                user.setName(name);
                user.setEmail(email);
                user.setRole(User.Role.USER);
                user.setStableId(stableId);

                // Generate password: Name + "123!"
                String rawPassword = name + "123!";
                user.setPassword(passwordEncoder.encode(rawPassword));
                user.setVisiblePassword(rawPassword); // Store visible password as requested

                userRepository.save(user);
                log.info("Created user: {} (ID: {})", name, stableId);
                logs.add("Created user: " + name + " (" + email + ")");
                added++;
            } catch (Exception e) {
                log.error("Error creating user {}: {}", name, e.getMessage());
                logs.add("Error creating user " + name + ": " + e.getMessage());
                errors++;
            }
        }

        result.put("status", "success");
        result.put("added", added);
        result.put("skipped", skipped);
        result.put("errors", errors);
        result.put("logs", logs);

        log.info("Sync completed. Added: {}, Skipped: {}, Errors: {}", added, skipped, errors);
        return result;
    }

    public List<Map<String, Object>> getAttendanceLogs() {
        List<Map<String, Object>> logs = new ArrayList<>();
        // Query database table creation ensures this table exists, or we query
        // sqlite_master
        String sql = "SELECT * FROM attendance ORDER BY timestamp DESC LIMIT 100";

        try (Connection conn = dataSource.getConnection();
                PreparedStatement stmt = conn.prepareStatement(sql);
                ResultSet rs = stmt.executeQuery()) {

            while (rs.next()) {
                Map<String, Object> log = new HashMap<>();
                log.put("id", rs.getInt("id"));
                log.put("person_name", rs.getString("person_name"));
                log.put("stable_id", rs.getInt("stable_id"));
                log.put("event_type", rs.getString("event_type"));
                log.put("timestamp", rs.getString("timestamp"));
                log.put("confidence", rs.getDouble("confidence"));
                logs.add(log);
            }
        } catch (Exception e) {
            log.error("Error fetching attendance logs: {}", e.getMessage());
        }
        return logs;
    }
}
