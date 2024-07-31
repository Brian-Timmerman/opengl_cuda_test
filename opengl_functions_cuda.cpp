#include "utilities_cuda.hpp"

const unsigned int SCR_WIDTH = 1920;
const unsigned int SCR_HEIGHT = 1080;
const unsigned int GRID_WIDTH = 50;
const unsigned int GRID_HEIGHT = 50;
const unsigned int NUM_SQUARES = GRID_WIDTH * GRID_HEIGHT;
const unsigned long long MAX_COUNT = 5000000ULL;

const char* vertexShaderSource = R"(
    #version 330 core
    layout (location = 0) in vec2 aPos;
    uniform mat4 model;
    uniform float intensity;
    out float fragIntensity;
    void main()
    {
        gl_Position = model * vec4(aPos.x, aPos.y, 0.0, 1.0);
        fragIntensity = intensity;
    }
)";

const char* fragmentShaderSource = R"(
    #version 330 core
    in float fragIntensity;
    out vec4 FragColor;
    void main()
    {
        vec3 color = vec3(0.0, 1.0, 0.0) * fragIntensity; // Vibrant green
        FragColor = vec4(color, 1.0);
    }
)";

void framebufferSizeCallback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

GLFWwindow* initializeOpenGL() {
    if (!glfwInit()) {
        std::cout << "Failed to initialize GLFW" << std::endl;
        return nullptr;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "OpenGL Squares", NULL, NULL);
    if (!window) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return nullptr;
    }

    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);

    if (glewInit() != GLEW_OK) {
        std::cout << "Failed to initialize GLEW" << std::endl;
        return nullptr;
    }

    return window;
}

void renderFrame(GLuint vbo, GLuint shaderProgram, GLuint VAO) {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shaderProgram);
    glBindVertexArray(VAO);
    
    // Bind the CUDA-interop buffer
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    float spacing = 0.02f;  // Reduced spacing
    float totalWidth = (GRID_WIDTH - 1) * (1 + spacing);
    float totalHeight = (GRID_HEIGHT - 1) * (1 + spacing);

    for (int i = 0; i < GRID_WIDTH; i++) {
        for (int j = 0; j < GRID_HEIGHT; j++) {
            int index = i * GRID_HEIGHT + j;
            
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::translate(model, glm::vec3(
                (i * (1 + spacing) - totalWidth / 2) / (GRID_WIDTH / 2.0f),
                (j * (1 + spacing) - totalHeight / 2) / (GRID_HEIGHT / 2.0f),
                0.0f
            ));
            model = glm::scale(model, glm::vec3(1.0f / GRID_WIDTH, 1.0f / GRID_HEIGHT, 1.0f));
            
            unsigned int modelLoc = glGetUniformLocation(shaderProgram, "model");
            glUniformMatrix4fv(modelLoc, 1, GL_FALSE, glm::value_ptr(model));
            
            unsigned long long counter;
            glGetBufferSubData(GL_ARRAY_BUFFER, index * sizeof(unsigned long long), sizeof(unsigned long long), &counter);
            
            float intensity = static_cast<float>(counter) / MAX_COUNT;
            unsigned int intensityLoc = glGetUniformLocation(shaderProgram, "intensity");
            glUniform1f(intensityLoc, intensity);
            
            glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        }
    }
}