static int compile_kernel_file(GPUEnv *gpu_env, const char *build_options)

{

    cl_int status;

    char *temp, *source_str = NULL;

    size_t source_str_len = 0;

    int i, ret = 0;



    for (i = 0; i < gpu_env->kernel_code_count; i++) {

        if (!gpu_env->kernel_code[i].is_compiled)

            source_str_len += strlen(gpu_env->kernel_code[i].kernel_string);

    }

    if (!source_str_len) {

        return 0;

    }

    source_str = av_mallocz(source_str_len + 1);

    if (!source_str) {

        return AVERROR(ENOMEM);

    }

    temp = source_str;

    for (i = 0; i < gpu_env->kernel_code_count; i++) {

        if (!gpu_env->kernel_code[i].is_compiled) {

            memcpy(temp, gpu_env->kernel_code[i].kernel_string,

                        strlen(gpu_env->kernel_code[i].kernel_string));

            gpu_env->kernel_code[i].is_compiled = 1;

            temp += strlen(gpu_env->kernel_code[i].kernel_string);

        }

    }

    /* create a CL program using the kernel source */

    gpu_env->programs[gpu_env->program_count] = clCreateProgramWithSource(gpu_env->context,

                                                           1, (const char **)(&source_str),

                                                                   &source_str_len, &status);

    if(status != CL_SUCCESS) {

        av_log(&openclutils, AV_LOG_ERROR, "Could not create OpenCL program with source code: %s\n",

               opencl_errstr(status));

        ret = AVERROR_EXTERNAL;

        goto end;

    }

    if (!gpu_env->programs[gpu_env->program_count]) {

        av_log(&openclutils, AV_LOG_ERROR, "Created program is NULL\n");

        ret = AVERROR_EXTERNAL;

        goto end;

    }

    i = 0;

    if (gpu_env->usr_spec_dev_info.dev_idx >= 0)

        i = gpu_env->usr_spec_dev_info.dev_idx;

    /* create a cl program executable for all the devices specified */

    if (!gpu_env->is_user_created)

        status = clBuildProgram(gpu_env->programs[gpu_env->program_count], 1, &gpu_env->device_ids[i],

                                build_options, NULL, NULL);

    else

        status = clBuildProgram(gpu_env->programs[gpu_env->program_count], 1, &(gpu_env->device_id),

                                 build_options, NULL, NULL);



    if (status != CL_SUCCESS) {

        av_log(&openclutils, AV_LOG_ERROR, "Could not compile OpenCL kernel: %s\n", opencl_errstr(status));

        ret = AVERROR_EXTERNAL;

        goto end;

    }

    gpu_env->program_count++;

end:

    av_free(source_str);

    return ret;

}
