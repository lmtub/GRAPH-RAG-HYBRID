static int init_opencl_env(GPUEnv *gpu_env, AVOpenCLExternalEnv *ext_opencl_env)

{

    size_t device_length;

    cl_int status;

    cl_uint num_platforms, num_devices;

    cl_platform_id *platform_ids = NULL;

    cl_context_properties cps[3];

    char platform_name[100];

    int i, ret = 0;

    cl_device_type device_type[] = {CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_DEFAULT};

    if (ext_opencl_env) {

        if (gpu_env->is_user_created)

            return 0;

        gpu_env->platform_id     = ext_opencl_env->platform_id;

        gpu_env->is_user_created = 1;

        gpu_env->command_queue   = ext_opencl_env->command_queue;

        gpu_env->context         = ext_opencl_env->context;

        gpu_env->device_ids      = ext_opencl_env->device_ids;

        gpu_env->device_id       = ext_opencl_env->device_id;

        gpu_env->device_type     = ext_opencl_env->device_type;

    } else {

        if (!gpu_env->is_user_created) {

            status = clGetPlatformIDs(0, NULL, &num_platforms);

            if (status != CL_SUCCESS) {

                av_log(&openclutils, AV_LOG_ERROR, "Could not get OpenCL platform ids: %s\n", opencl_errstr(status));

                return AVERROR_EXTERNAL;

            }

            if (gpu_env->usr_spec_dev_info.platform_idx >= 0) {

                if (num_platforms < gpu_env->usr_spec_dev_info.platform_idx + 1) {

                    av_log(&openclutils, AV_LOG_ERROR, "User set platform index not exist\n");

                    return AVERROR(EINVAL);

                }

            }

            if (num_platforms > 0) {

                platform_ids = av_mallocz(num_platforms * sizeof(cl_platform_id));

                if (!platform_ids) {

                    ret = AVERROR(ENOMEM);

                    goto end;

                }

                status = clGetPlatformIDs(num_platforms, platform_ids, NULL);

                if (status != CL_SUCCESS) {

                    av_log(&openclutils, AV_LOG_ERROR, "Could not get OpenCL platform ids: %s\n", opencl_errstr(status));

                    ret = AVERROR_EXTERNAL;

                    goto end;

                }

                i = 0;

                if (gpu_env->usr_spec_dev_info.platform_idx >= 0) {

                    i = gpu_env->usr_spec_dev_info.platform_idx;

                }

                while (i < num_platforms) {

                    status = clGetPlatformInfo(platform_ids[i], CL_PLATFORM_VENDOR,

                                               sizeof(platform_name), platform_name,

                                               NULL);



                    if (status != CL_SUCCESS) {

                        av_log(&openclutils, AV_LOG_ERROR, "Could not get OpenCL platform info: %s\n", opencl_errstr(status));

                        ret = AVERROR_EXTERNAL;

                        goto end;

                    }

                    gpu_env->platform_id = platform_ids[i];

                    status = clGetDeviceIDs(gpu_env->platform_id, CL_DEVICE_TYPE_GPU,

                                            0, NULL, &num_devices);

                    if (status != CL_SUCCESS) {

                        av_log(&openclutils, AV_LOG_ERROR, "Could not get OpenCL device number:%s\n", opencl_errstr(status));

                        ret = AVERROR_EXTERNAL;

                        goto end;

                    }

                    if (num_devices == 0) {

                        //find CPU device

                        status = clGetDeviceIDs(gpu_env->platform_id, CL_DEVICE_TYPE_CPU,

                                             0, NULL, &num_devices);

                    }

                    if (status != CL_SUCCESS) {

                        av_log(&openclutils, AV_LOG_ERROR, "Could not get OpenCL device ids: %s\n", opencl_errstr(status));

                        ret = AVERROR(EINVAL);

                        goto end;

                    }

                    if (num_devices)

                       break;

                    if (gpu_env->usr_spec_dev_info.platform_idx >= 0) {

                        av_log(&openclutils, AV_LOG_ERROR, "Device number of user set platform is 0\n");

                        ret = AVERROR_EXTERNAL;

                        goto end;

                    }

                    i++;



                }

            }

            if (!gpu_env->platform_id) {

                av_log(&openclutils, AV_LOG_ERROR, "Could not get OpenCL platforms\n");

                ret = AVERROR_EXTERNAL;

                goto end;

            }

            if (gpu_env->usr_spec_dev_info.dev_idx >= 0) {

                if (num_devices < gpu_env->usr_spec_dev_info.dev_idx + 1) {

                    av_log(&openclutils, AV_LOG_ERROR, "Could not get OpenCL device idx in the user set platform\n");

                    ret = AVERROR(EINVAL);

                    goto end;

                }

            }



           /*

                    * Use available platform.

                    */



            av_log(&openclutils, AV_LOG_VERBOSE, "Platform Name: %s\n", platform_name);

            cps[0] = CL_CONTEXT_PLATFORM;

            cps[1] = (cl_context_properties)gpu_env->platform_id;

            cps[2] = 0;

            /* Check for GPU. */



            for (i = 0; i < sizeof(device_type); i++) {

                gpu_env->device_type = device_type[i];

                gpu_env->context     = clCreateContextFromType(cps, gpu_env->device_type,

                                                               NULL, NULL, &status);

                if (status != CL_SUCCESS) {

                    av_log(&openclutils, AV_LOG_ERROR, "Could not get OpenCL context from device type: %s\n", opencl_errstr(status));

                    ret = AVERROR_EXTERNAL;

                    goto end;

                }

                if (gpu_env->context)

                    break;

            }

            if (!gpu_env->context) {

                av_log(&openclutils, AV_LOG_ERROR, "Could not get OpenCL context from device type\n");

                ret = AVERROR_EXTERNAL;

                goto end;

            }

            /* Detect OpenCL devices. */

            /* First, get the size of device list data */

            status = clGetContextInfo(gpu_env->context, CL_CONTEXT_DEVICES,

                                      0, NULL, &device_length);

            if (status != CL_SUCCESS) {

                av_log(&openclutils, AV_LOG_ERROR, "Could not get OpenCL device length: %s\n", opencl_errstr(status));

                ret = AVERROR_EXTERNAL;

                goto end;

            }

            if (device_length == 0) {

                av_log(&openclutils, AV_LOG_ERROR, "Could not get OpenCL device length\n");

                ret = AVERROR_EXTERNAL;

                goto end;

            }

            /* Now allocate memory for device list based on the size we got earlier */

            gpu_env->device_ids = av_mallocz(device_length);

            if (!gpu_env->device_ids) {

                ret = AVERROR(ENOMEM);

                goto end;

            }

            /* Now, get the device list data */

            status = clGetContextInfo(gpu_env->context, CL_CONTEXT_DEVICES, device_length,

                                      gpu_env->device_ids, NULL);

            if (status != CL_SUCCESS) {

                av_log(&openclutils, AV_LOG_ERROR, "Could not get OpenCL context info: %s\n", opencl_errstr(status));

                ret = AVERROR_EXTERNAL;

                goto end;

            }

            /* Create OpenCL command queue. */

            i = 0;

            if (gpu_env->usr_spec_dev_info.dev_idx >= 0) {

                i = gpu_env->usr_spec_dev_info.dev_idx;

            }

            gpu_env->command_queue = clCreateCommandQueue(gpu_env->context, gpu_env->device_ids[i],

                                                          0, &status);

            if (status != CL_SUCCESS) {

                av_log(&openclutils, AV_LOG_ERROR, "Could not create OpenCL command queue: %s\n", opencl_errstr(status));

                ret = AVERROR_EXTERNAL;

                goto end;

            }

        }

    }

end:

    av_free(platform_ids);

    return ret;

}
