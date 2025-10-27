static JavaVM *get_java_vm(const char *name, void *log_ctx)

{

    JavaVM *vm = NULL;

    jsize nb_vm = 0;



    void *handle = NULL;

    jint (*get_created_java_vms) (JavaVM ** vmBuf, jsize bufLen, jsize *nVMs) = NULL;



    handle = dlopen(name, RTLD_LOCAL);

    if (!handle) {

        return NULL;

    }



    get_created_java_vms = (jint (*)(JavaVM **, jsize, jsize *)) dlsym(handle, "JNI_GetCreatedJavaVMs");

    if (!get_created_java_vms) {

        av_log(log_ctx, AV_LOG_ERROR, "Could not find JNI_GetCreatedJavaVMs symbol in library '%s'\n", name);

        goto done;

    }



    if (get_created_java_vms(&vm, 1, &nb_vm) != JNI_OK) {

        av_log(log_ctx, AV_LOG_ERROR, "Could not get created Java virtual machines\n");

        goto done;

    }



done:

    if (handle) {

        dlclose(handle);

    }



    return vm;

}
