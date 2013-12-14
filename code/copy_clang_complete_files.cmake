function(copy_clang_complete_files cur_dir)
    message("### Copying .clang_complete files: ###")
    file(GLOB_RECURSE items  RELATIVE ${cur_dir} ${cur_dir}/.clang_complete)
    # Get relative paths , attach them to source dir 
    # and copy .clang_complete files if possible
    foreach(item ${items})   
        string(REPLACE .clang_complete  "" rel_path ${item})
        if(IS_DIRECTORY ${SOURCE_DIR}/${rel_path})
            file(COPY ${cur_dir}/${item} DESTINATION ${SOURCE_DIR}/${rel_path})
            message("${cur_dir}/${item} ---> ${SOURCE_DIR}/${rel_path}")
        endif()
    endforeach(item) 
endfunction(copy_clang_complete_files)


copy_clang_complete_files(${CMAKE_CURRENT_BINARY_DIR})
