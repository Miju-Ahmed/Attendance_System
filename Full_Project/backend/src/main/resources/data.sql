-- Admin user seed data for SQLite
-- Password: Miju (BCrypt hash with strength 12)
-- This will only insert if the email doesn't already exist
INSERT OR IGNORE INTO users (email, name, password, role, stable_id, created_at, updated_at)
VALUES (
    'miju.ch7@gmail.com',
    'Admin User',
    '$2a$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY5GyYzOtcWxZK6',
    'ADMIN',
    1,
    datetime('now'),
    datetime('now')
);

