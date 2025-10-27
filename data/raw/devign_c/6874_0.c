int av_opencl_init(AVDictionary *options, AVOpenCLExternalEnv *ext_opencl_env)

{

    int ret = 0;

    AVDictionaryEntry *opt_build_entry;

    AVDictionaryEntry *opt_platform_entry;

    AVDictionaryEntry *opt_device_entry;

    LOCK_OPENCL

    if (!gpu_env.init_count) {

        opt_platform_entry = av_dict_get(options, "platform_idx", NULL, 0);

        opt_device_entry   = av_dict_get(options, "device_idx", NULL, 0);

        /*initialize devices, context, command_queue*/

        gpu_env.usr_spec_dev_info.platform_idx = -1;

        gpu_env.usr_spec_dev_info.dev_idx = -1;

        if (opt_platform_entry) {

            gpu_env.usr_spec_dev_info.platform_idx = strtol(opt_platform_entry->value, NULL, 10);

        }

        if (opt_device_entry) {

            gpu_env.usr_spec_dev_info.dev_idx = strtol(opt_device_entry->value, NULL, 10);

        }

        ret = init_opencl_env(&gpu_env, ext_opencl_env);

        if (ret < 0)

            goto end;

    }

    /*initialize program, kernel_name, kernel_count*/

    opt_build_entry = av_dict_get(options, "build_options", NULL, 0);

    if (opt_build_entry)

        ret = compile_kernel_file(&gpu_env, opt_build_entry->value);

    else

        ret = compile_kernel_file(&gpu_env, NULL);

    if (ret < 0)

        goto end;

    av_assert1(gpu_env.kernel_code_count > 0);

    gpu_env.init_count++;



end:

    UNLOCK_OPENCL

    return ret;

}
