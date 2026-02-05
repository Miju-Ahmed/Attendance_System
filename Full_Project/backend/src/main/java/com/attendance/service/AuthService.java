package com.attendance.service;

import com.attendance.dto.JwtResponse;
import com.attendance.dto.LoginRequest;
import com.attendance.dto.RegisterRequest;
import com.attendance.entity.User;
import com.attendance.repository.UserRepository;
import com.attendance.security.JwtTokenProvider;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.security.authentication.AuthenticationManager;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

@Service
@RequiredArgsConstructor
@Slf4j
public class AuthService {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;
    private final AuthenticationManager authenticationManager;
    private final JwtTokenProvider tokenProvider;

    @Transactional
    public JwtResponse login(LoginRequest loginRequest) {
        log.info("Login attempt for email: {}", loginRequest.getEmail());

        try {
            Authentication authentication = authenticationManager.authenticate(
                    new UsernamePasswordAuthenticationToken(
                            loginRequest.getEmail(),
                            loginRequest.getPassword()));

            SecurityContextHolder.getContext().setAuthentication(authentication);
            String jwt = tokenProvider.generateToken(authentication);

            User user = userRepository.findByEmail(loginRequest.getEmail())
                    .orElseThrow(() -> new RuntimeException("User not found"));

            log.info("Login successful for user: {} with role: {}", user.getEmail(), user.getRole());
            return new JwtResponse(jwt, user.getId(), user.getName(), user.getEmail(), user.getRole().name());
        } catch (Exception e) {
            log.error("Login failed for email: {}. Error: {}", loginRequest.getEmail(), e.getMessage());
            throw e;
        }
    }

    @Transactional
    public User register(RegisterRequest registerRequest) {
        log.info("Registration attempt for email: {}", registerRequest.getEmail());

        if (userRepository.existsByEmail(registerRequest.getEmail())) {
            log.warn("Registration failed - email already exists: {}", registerRequest.getEmail());
            throw new RuntimeException("Email already exists");
        }

        User user = new User();
        user.setName(registerRequest.getName());
        user.setEmail(registerRequest.getEmail());
        user.setPassword(passwordEncoder.encode(registerRequest.getPassword()));
        user.setRole(User.Role.USER);
        user.setPhone(registerRequest.getPhone());
        user.setAddress(registerRequest.getAddress());

        // Assign stable_id for face recognition
        Integer maxStableId = userRepository.findAll().stream()
                .map(User::getStableId)
                .filter(id -> id != null)
                .max(Integer::compareTo)
                .orElse(1);
        user.setStableId(maxStableId + 1);

        User savedUser = userRepository.save(user);
        log.info("User registered successfully: {} with stable_id: {}", savedUser.getEmail(), savedUser.getStableId());
        return savedUser;
    }

    public User getCurrentUser() {
        Authentication authentication = SecurityContextHolder.getContext().getAuthentication();
        String email = authentication.getName();
        return userRepository.findByEmail(email)
                .orElseThrow(() -> new RuntimeException("User not found"));
    }
}
