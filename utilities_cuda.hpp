#pragma once
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>

extern const unsigned int SCR_WIDTH;
extern const unsigned int SCR_HEIGHT;
extern const unsigned int GRID_WIDTH;
extern const unsigned int GRID_HEIGHT;
extern const unsigned int NUM_SQUARES;
extern const unsigned long long MAX_COUNT;
extern std::atomic<bool> shouldExit;
extern const char* vertexShaderSource;
extern const char* fragmentShaderSource;

using high_res_clock = std::chrono::high_resolution_clock;

extern void framebufferSizeCallback(GLFWwindow* window, int width, int height);
extern GLFWwindow* initializeOpenGL();
extern void renderFrame(GLuint vbo, GLuint shaderProgram, GLuint VAO);