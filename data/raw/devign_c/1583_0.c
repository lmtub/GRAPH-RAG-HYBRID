static int check_jni_invocation(void *log_ctx)

{

    int ret = AVERROR_EXTERNAL;

    void *handle = NULL;

    void **jni_invocation = NULL;



    handle = dlopen(NULL, RTLD_LOCAL);

    if (!handle) {

        goto done;

    }



    jni_invocation = (void **)dlsym(handle, "_ZN13JniInvocation15jni_invocation_E");

    if (!jni_invocation) {

        av_log(log_ctx, AV_LOG_ERROR, "Could not find JniInvocation::jni_invocation_ symbol\n");

        goto done;

    }



    ret = !(jni_invocation != NULL && *jni_invocation != NULL);



done:

    if (handle) {

        dlclose(handle);

    }



    return ret;

}
