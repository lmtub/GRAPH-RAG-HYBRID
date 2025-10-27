JNIEnv *ff_jni_attach_env(int *attached, void *log_ctx)

{

    int ret = 0;

    JNIEnv *env = NULL;



    *attached = 0;



    pthread_mutex_lock(&lock);

    if (java_vm == NULL && (java_vm = av_jni_get_java_vm(log_ctx)) == NULL) {



        av_log(log_ctx, AV_LOG_INFO, "Retrieving current Java virtual machine using Android JniInvocation wrapper\n");

        if (check_jni_invocation(log_ctx) == 0) {

            if ((java_vm = get_java_vm(NULL, log_ctx)) != NULL ||

                (java_vm = get_java_vm("libdvm.so", log_ctx)) != NULL ||

                (java_vm = get_java_vm("libart.so", log_ctx)) != NULL) {

                av_log(log_ctx, AV_LOG_INFO, "Found Java virtual machine using Android JniInvocation wrapper\n");

            }

        }

    }

    pthread_mutex_unlock(&lock);



    if (!java_vm) {

        av_log(log_ctx, AV_LOG_ERROR, "Could not retrieve a Java virtual machine\n");

        return NULL;

    }



    ret = (*java_vm)->GetEnv(java_vm, (void **)&env, JNI_VERSION_1_6);

    switch(ret) {

    case JNI_EDETACHED:

        if ((*java_vm)->AttachCurrentThread(java_vm, &env, NULL) != 0) {

            av_log(log_ctx, AV_LOG_ERROR, "Failed to attach the JNI environment to the current thread\n");

            env = NULL;

        } else {

            *attached = 1;

        }

        break;

    case JNI_OK:

        break;

    case JNI_EVERSION:

        av_log(log_ctx, AV_LOG_ERROR, "The specified JNI version is not supported\n");

        break;

    default:

        av_log(log_ctx, AV_LOG_ERROR, "Failed to get the JNI environment attached to this thread");

        break;

    }



    return env;

}
