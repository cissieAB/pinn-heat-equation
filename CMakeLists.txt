
cmake_minimum_required(VERSION 3.9)

project(pinn)

set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)

set(PINN_SOURCES pinn_diffusion_pde_solver.cpp)
add_executable(${PROJECT_NAME} ${PINN_SOURCES})

target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES} )

