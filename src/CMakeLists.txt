
find_package(Torch REQUIRED)

set(PINN_DIFFUSION_PDE_SOLVER_SOURCES pinn_diffusion_pde_solver.cpp)
add_executable(phasm-example-pinn-pdesolver ${PINN_DIFFUSION_PDE_SOLVER_SOURCES})
target_include_directories(phasm-example-pinn-pdesolver PRIVATE
#        ${CMAKE_CURRENT_SOURCE_DIR}/../../surrogate/include
#        ${CMAKE_CURRENT_SOURCE_DIR}/../../memtrace/include
        )
target_link_libraries(phasm-example-pinn-pdesolver ${TORCH_LIBRARIES} )
install(TARGETS phasm-example-pinn-pdesolver DESTINATION bin)
