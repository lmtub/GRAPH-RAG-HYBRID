int ff_jni_exception_get_summary(JNIEnv *env, jthrowable exception, char **error, void *log_ctx)

{

    int ret = 0;



    AVBPrint bp;



    char *name = NULL;

    char *message = NULL;



    jclass class_class = NULL;

    jmethodID get_name_id = NULL;



    jclass exception_class = NULL;

    jmethodID get_message_id = NULL;



    jstring string;



    av_bprint_init(&bp, 0, AV_BPRINT_SIZE_AUTOMATIC);



    exception_class = (*env)->GetObjectClass(env, exception);

    if ((*env)->ExceptionCheck(env)) {

        (*env)->ExceptionClear(env);

        av_log(log_ctx, AV_LOG_ERROR, "Could not find Throwable class\n");

        ret = AVERROR_EXTERNAL;

        goto done;

    }



    class_class = (*env)->GetObjectClass(env, exception_class);

    if ((*env)->ExceptionCheck(env)) {

        (*env)->ExceptionClear(env);

        av_log(log_ctx, AV_LOG_ERROR, "Could not find Throwable class's class\n");

        ret = AVERROR_EXTERNAL;

        goto done;

    }



    get_name_id = (*env)->GetMethodID(env, class_class, "getName", "()Ljava/lang/String;");

    if ((*env)->ExceptionCheck(env)) {

        (*env)->ExceptionClear(env);

        av_log(log_ctx, AV_LOG_ERROR, "Could not find method Class.getName()\n");

        ret = AVERROR_EXTERNAL;

        goto done;

    }



    string = (*env)->CallObjectMethod(env, exception_class, get_name_id);

    if ((*env)->ExceptionCheck(env)) {

        (*env)->ExceptionClear(env);

        av_log(log_ctx, AV_LOG_ERROR, "Class.getName() threw an exception\n");

        ret = AVERROR_EXTERNAL;

        goto done;

    }



    if (string) {

        name = ff_jni_jstring_to_utf_chars(env, string, log_ctx);

        (*env)->DeleteLocalRef(env, string);

        string = NULL;

    }



    get_message_id = (*env)->GetMethodID(env, exception_class, "getMessage", "()Ljava/lang/String;");

    if ((*env)->ExceptionCheck(env)) {

        (*env)->ExceptionClear(env);

        av_log(log_ctx, AV_LOG_ERROR, "Could not find method java/lang/Throwable.getMessage()\n");

        ret = AVERROR_EXTERNAL;

        goto done;

    }



    string = (*env)->CallObjectMethod(env, exception, get_message_id);

    if ((*env)->ExceptionCheck(env)) {

        (*env)->ExceptionClear(env);

        av_log(log_ctx, AV_LOG_ERROR, "Throwable.getMessage() threw an exception\n");

        ret = AVERROR_EXTERNAL;

        goto done;

    }



    if (string) {

        message = ff_jni_jstring_to_utf_chars(env, string, log_ctx);

        (*env)->DeleteLocalRef(env, string);

        string = NULL;

    }



    if (name && message) {

        av_bprintf(&bp, "%s: %s", name, message);

    } else if (name && !message) {

        av_bprintf(&bp, "%s occurred", name);

    } else if (!name && message) {

        av_bprintf(&bp, "Exception: %s", message);

    } else {

        av_log(log_ctx, AV_LOG_WARNING, "Could not retreive exception name and message\n");

        av_bprintf(&bp, "Exception occurred");

    }



    ret = av_bprint_finalize(&bp, error);

done:



    av_free(name);

    av_free(message);



    if (class_class) {

        (*env)->DeleteLocalRef(env, class_class);

    }



    if (exception_class) {

        (*env)->DeleteLocalRef(env, exception_class);

    }



    if (string) {

        (*env)->DeleteLocalRef(env, string);

    }



    return ret;

}
