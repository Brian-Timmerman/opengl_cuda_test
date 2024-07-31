#include "utilities_cuda.hpp"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>

__global__ void incrementCounters(unsigned long long* counters, int num_squares, unsigned long long max_count) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_squares) {
        atomicAdd(&counters[idx], 1ULL);
    }
}

int main() {
    GLFWwindow* window = initializeOpenGL();
    if (!window) {
        return -1;
    }
    GLuint VBO, VAO, EBO, shaderProgram;
    // Set up VAO
    glGenVertexArrays(1, &VAO);
    glBindVertexArray(VAO);

    // Set up VBO
    glGenBuffers(1, &VBO);
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    float vertices[] = {
        -0.5f,  0.5f,
        0.5f,  0.5f,
        0.5f, -0.5f,
        -0.5f, -0.5f
    };
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Set up EBO
    glGenBuffers(1, &EBO);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    unsigned int indices[] = {
        0, 1, 2,
        2, 3, 0
    };
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // Set up vertex attributes
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Create and compile shaders
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    // Create shader program
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Clean up
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    GLuint vbo;
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, NUM_SQUARES * sizeof(unsigned long long), NULL, GL_DYNAMIC_DRAW);

    cudaGraphicsResource* cuda_vbo_resource;
    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsWriteDiscard);

    int threadsPerBlock = 256;
    int blocksPerGrid = (NUM_SQUARES + threadsPerBlock - 1) / threadsPerBlock;

    while (!glfwWindowShouldClose(window)) {
        unsigned long long* d_counters;
        size_t num_bytes;
        cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_counters, &num_bytes, cuda_vbo_resource);

        incrementCounters<<<blocksPerGrid, threadsPerBlock>>>(d_counters, NUM_SQUARES, MAX_COUNT);

        cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
        cudaDeviceSynchronize();  // Add this line
        renderFrame(vbo, shaderProgram, VAO);

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaGraphicsUnregisterResource(cuda_vbo_resource);
    glDeleteBuffers(1, &vbo);
    glfwTerminate();

    return 0;
}