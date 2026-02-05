package com.attendance.service;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.concurrent.CompletableFuture;

@Service
public class VideoProcessingService {

    private static final Logger log = LoggerFactory.getLogger(VideoProcessingService.class);
    // Path to the python script relative to backend execution directory or absolute
    private static final String PYTHON_SCRIPT_PATH = "../AI_Portion/attendance_efficientnetdet.py";

    @Async
    public CompletableFuture<String> processVideo(String videoFilePath, String outputFilePath, boolean isInteractive) {
        log.info("Starting video processing for file: {}", videoFilePath);
        StringBuilder output = new StringBuilder();

        try {
            // Validate file exists, unless it is "0" (Live Camera)
            if (!"0".equals(videoFilePath)) {
                File file = new File(videoFilePath);
                if (!file.exists()) {
                    throw new RuntimeException("Video file not found: " + videoFilePath);
                }
            }

            // Construct command: python3 ../AI_Portion/attendance_efficientnetdet.py
            // --source ...
            java.util.List<String> command = new java.util.ArrayList<>();
            command.add("python3");
            command.add(PYTHON_SCRIPT_PATH);
            command.add("--source");
            command.add(videoFilePath);
            command.add("--device");
            command.add("cpu");

            if (outputFilePath != null) {
                command.add("--output");
                command.add(outputFilePath);
            }

            if (!isInteractive) {
                command.add("--no-interaction");
                command.add("--skip-display");
            }

            ProcessBuilder pb = new ProcessBuilder(command);

            // Set working directory to AI_Portion so relative paths in script work
            pb.directory(new File("../AI_Portion"));
            pb.redirectErrorStream(true);

            Process process = pb.start();

            // Read output
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    log.debug("Python Output: {}", line);
                    output.append(line).append("\n");
                }
            }

            int exitCode = process.waitFor();
            if (exitCode == 0) {
                log.info("Video processing completed successfully.");
                return CompletableFuture.completedFuture(outputFilePath != null ? outputFilePath : "Success");
            } else {
                log.error("Video processing failed with exit code: {}", exitCode);
                return CompletableFuture
                        .failedFuture(new RuntimeException("Processing failed with exit code " + exitCode));
            }

        } catch (Exception e) {
            log.error("Error executing video processing: {}", e.getMessage(), e);
            return CompletableFuture.failedFuture(e);
        }
    }
}
