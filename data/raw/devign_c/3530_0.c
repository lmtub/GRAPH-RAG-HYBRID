void av_opencl_buffer_release(cl_mem *cl_buf)

{

    cl_int status = 0;

    if (!cl_buf)

        return;

    status = clReleaseMemObject(*cl_buf);

    if (status != CL_SUCCESS) {

        av_log(&openclutils, AV_LOG_ERROR, "Could not release OpenCL buffer: %s\n", opencl_errstr(status));

    }

    memset(cl_buf, 0, sizeof(*cl_buf));

}
