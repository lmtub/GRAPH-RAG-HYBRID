int av_opencl_register_kernel_code(const char *kernel_code)

{

    int i, ret = 0;

    LOCK_OPENCL;

    if (gpu_env.kernel_code_count >= MAX_KERNEL_CODE_NUM) {

        av_log(&openclutils, AV_LOG_ERROR,

         "Could not register kernel code, maximum number of registered kernel code %d already reached\n",

         MAX_KERNEL_CODE_NUM);

        ret = AVERROR(EINVAL);

        goto end;

    }

    for (i = 0; i < gpu_env.kernel_code_count; i++) {

        if (gpu_env.kernel_code[i].kernel_string == kernel_code) {

            av_log(&openclutils, AV_LOG_WARNING, "Same kernel code has been registered\n");

            goto end;

        }

    }

    gpu_env.kernel_code[gpu_env.kernel_code_count].kernel_string = kernel_code;

    gpu_env.kernel_code[gpu_env.kernel_code_count].is_compiled = 0;

    gpu_env.kernel_code_count++;

end:

    UNLOCK_OPENCL;

    return ret;

}
