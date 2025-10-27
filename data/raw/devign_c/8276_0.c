AVOpenCLExternalEnv *av_opencl_alloc_external_env(void)

{

    AVOpenCLExternalEnv *ext = av_mallocz(sizeof(AVOpenCLExternalEnv));

    if (!ext) {

        av_log(&openclutils, AV_LOG_ERROR,

         "Could not malloc external opencl environment data space\n");

    }

    return ext;

}
