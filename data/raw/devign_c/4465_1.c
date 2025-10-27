static int64_t run_opencl_bench(AVOpenCLExternalEnv *ext_opencl_env)

{

    int i, arg = 0, width = 1920, height = 1088;

    int64_t start, ret = 0;

    cl_int status;

    size_t kernel_len;

    char *inbuf;

    int *mask;

    int buf_size = width * height * sizeof(char);

    int mask_size = sizeof(uint32_t) * 128;



    cl_mem cl_mask, cl_inbuf, cl_outbuf;

    cl_kernel kernel = NULL;

    cl_program program = NULL;

    size_t local_work_size_2d[2] = {16, 16};

    size_t global_work_size_2d[2] = {(size_t)width, (size_t)height};



    if (!(inbuf = av_malloc(buf_size)) || !(mask = av_malloc(mask_size))) {

        av_log(NULL, AV_LOG_ERROR, "Out of memory\n");

        ret = AVERROR(ENOMEM);

        goto end;

    }

    fill_rand_int((int*)inbuf, buf_size/4);

    fill_rand_int(mask, mask_size/4);



    CREATEBUF(cl_mask, CL_MEM_READ_ONLY, mask_size);

    CREATEBUF(cl_inbuf, CL_MEM_READ_ONLY, buf_size);

    CREATEBUF(cl_outbuf, CL_MEM_READ_WRITE, buf_size);



    kernel_len = strlen(ocl_bench_source);

    program = clCreateProgramWithSource(ext_opencl_env->context, 1, &ocl_bench_source,

                                        &kernel_len, &status);

    if (status != CL_SUCCESS || !program) {

        av_log(NULL, AV_LOG_ERROR, "OpenCL unable to create benchmark program\n");

        ret = AVERROR_EXTERNAL;

        goto end;

    }

    status = clBuildProgram(program, 1, &(ext_opencl_env->device_id), NULL, NULL, NULL);

    if (status != CL_SUCCESS) {

        av_log(NULL, AV_LOG_ERROR, "OpenCL unable to build benchmark program\n");

        ret = AVERROR_EXTERNAL;

        goto end;

    }

    kernel = clCreateKernel(program, "unsharp_bench", &status);

    if (status != CL_SUCCESS) {

        av_log(NULL, AV_LOG_ERROR, "OpenCL unable to create benchmark kernel\n");

        ret = AVERROR_EXTERNAL;

        goto end;

    }



    OCLCHECK(clEnqueueWriteBuffer, ext_opencl_env->command_queue, cl_inbuf, CL_TRUE, 0,

             buf_size, inbuf, 0, NULL, NULL);

    OCLCHECK(clEnqueueWriteBuffer, ext_opencl_env->command_queue, cl_mask, CL_TRUE, 0,

             mask_size, mask, 0, NULL, NULL);

    OCLCHECK(clSetKernelArg, kernel, arg++, sizeof(cl_mem), &cl_inbuf);

    OCLCHECK(clSetKernelArg, kernel, arg++, sizeof(cl_mem), &cl_outbuf);

    OCLCHECK(clSetKernelArg, kernel, arg++, sizeof(cl_mem), &cl_mask);

    OCLCHECK(clSetKernelArg, kernel, arg++, sizeof(cl_int), &width);

    OCLCHECK(clSetKernelArg, kernel, arg++, sizeof(cl_int), &height);



    start = av_gettime_relative();

    for (i = 0; i < OPENCL_NB_ITER; i++)

        OCLCHECK(clEnqueueNDRangeKernel, ext_opencl_env->command_queue, kernel, 2, NULL,

                 global_work_size_2d, local_work_size_2d, 0, NULL, NULL);

    clFinish(ext_opencl_env->command_queue);

    ret = (av_gettime_relative() - start)/OPENCL_NB_ITER;

end:

    if (kernel)

        clReleaseKernel(kernel);

    if (program)

        clReleaseProgram(program);

    if (cl_inbuf)

        clReleaseMemObject(cl_inbuf);

    if (cl_outbuf)

        clReleaseMemObject(cl_outbuf);

    if (cl_mask)

        clReleaseMemObject(cl_mask);

    av_free(inbuf);

    av_free(mask);

    return ret;

}
