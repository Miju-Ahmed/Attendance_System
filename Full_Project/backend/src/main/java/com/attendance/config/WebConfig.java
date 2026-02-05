package com.attendance.config;

import org.springframework.context.annotation.Configuration;
import org.springframework.web.servlet.config.annotation.ResourceHandlerRegistry;
import org.springframework.web.servlet.config.annotation.WebMvcConfigurer;

@Configuration
public class WebConfig implements WebMvcConfigurer {

    @Override
    public void addResourceHandlers(ResourceHandlerRegistry registry) {
        // Map /uploads/** to the directory on disk
        // Note: 'file:../AI_Portion/uploads/' assumes running from backend dir
        registry.addResourceHandler("/uploads/**")
                .addResourceLocations("file:../AI_Portion/uploads/");
    }
}
