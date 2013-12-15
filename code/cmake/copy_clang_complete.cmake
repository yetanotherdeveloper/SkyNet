add_custom_target(copy_clang_complete 
        COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} -P ${CMAKE_CURRENT_LIST_DIR}/copy_clang_complete_files.cmake)

