#include "utilities_cuda.hpp"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <device_launch_parameters.h>



__global__ void updateIntensities(float* intensities, int num_squares) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_squares) {
        intensities[idx] = intensities[idx] + 0.001f;
    }
}

int main() {

    float* d_intensities;
    cudaMalloc(&d_intensities, GRID_WIDTH * GRID_HEIGHT * sizeof(float));

    // Initialize GLFW
    if (!glfwInit())
    {
        std::cout << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Configure GLFW
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    #ifdef __APPLE__
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    #endif

    // Create a window
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "OpenGL Squares", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    // Initialize GLEW
    if (glewInit() != GLEW_OK)
    {
        std::cout << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Vertex data for a single square
    float vertices[] = {
        0.5f,  0.5f,
        0.5f, -0.5f,
        -0.5f, -0.5f,
        -0.5f,  0.5f
    };

    unsigned int indices[] = {
        0, 1, 3,
        1, 2, 3
    };

    // Set up VAO and VBO
    unsigned int VAO, VBO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    // Create intensity VBO and register with CUDA
    GLuint intensityVBO;
    glGenBuffers(1, &intensityVBO);
    glBindBuffer(GL_ARRAY_BUFFER, intensityVBO);
    glBufferData(GL_ARRAY_BUFFER, GRID_WIDTH * GRID_HEIGHT * sizeof(float), NULL, GL_DYNAMIC_DRAW);

    cudaGraphicsResource* cuda_intensity_resource;
    cudaGraphicsGLRegisterBuffer(&cuda_intensity_resource, intensityVBO, cudaGraphicsMapFlagsWriteDiscard);

    // Create and compile shaders
    unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    // Create shader program
    unsigned int shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    unsigned int intensityLoc = glGetUniformLocation(shaderProgram, "intensity");
   // Render loop
    while(!glfwWindowShouldClose(window)) {
        // Clear the screen with black color
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Map OpenGL buffer to CUDA
        float* d_mapped_intensities;
        size_t num_bytes;
        cudaGraphicsMapResources(1, &cuda_intensity_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_mapped_intensities, &num_bytes, cuda_intensity_resource);

        // Update intensities using CUDA
        int threadsPerBlock = 256;
        int blocksPerGrid = (GRID_WIDTH * GRID_HEIGHT + threadsPerBlock - 1) / threadsPerBlock;
        updateIntensities<<<blocksPerGrid, threadsPerBlock>>>(d_mapped_intensities, GRID_WIDTH * GRID_HEIGHT);

        // Unmap buffer
        cudaGraphicsUnmapResources(1, &cuda_intensity_resource, 0);

        // Use the updated intensities in OpenGL
        glBindBuffer(GL_ARRAY_BUFFER, intensityVBO);
        glVertexAttribPointer(1, 1, GL_FLOAT, GL_FALSE, sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);

        // Use shader program
        glUseProgram(shaderProgram);

        // Draw squares
        glBindVertexArray(VAO);
        float spacing = 0.2f; // Space between squares
        float totalWidth = (GRID_WIDTH - 1) * (1 + spacing);
        float totalHeight = (GRID_HEIGHT - 1) * (1 + spacing);
        for (int i = 0; i < GRID_WIDTH; i++) {
            for (int j = 0; j < GRID_HEIGHT; j++) {
                int index = i * GRID_HEIGHT + j;
                
                glm::mat4 model = glm::mat4(1.0f);
                model = glm::translate(model, glm::vec3(
                    (i * (1 + spacing) - totalWidth / 2) / GRID_WIDTH,
                    (j * (1 + spacing) - totalHeight / 2) / GRID_HEIGHT,
                    0.0f
                ));
                model = glm::scale(model, glm::vec3(1.0f / (GRID_WIDTH * (1 + spacing)), 
                                                    1.0f / (GRID_HEIGHT * (1 + spacing)), 
                                                    1.0f));
                
                unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
                glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
                
                // Set the intensity for this square
                float intensity = 1.0;
                glUniform1f(intensityLoc, intensity);
                
                glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
            }
        }

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteProgram(shaderProgram);
    glfwTerminate();
    cudaFree(d_intensities);
    cudaGraphicsUnregisterResource(cuda_intensity_resource);
    return 0;
}