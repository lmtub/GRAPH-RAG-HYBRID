int ff_jni_init_jfields(JNIEnv *env, void *jfields, const struct FFJniField *jfields_mapping, int global, void *log_ctx)
{
    int i, ret = 0;
    jclass last_clazz = NULL;
    for (i = 0; jfields_mapping[i].name; i++) {
        int mandatory = jfields_mapping[i].mandatory;
        enum FFJniFieldType type = jfields_mapping[i].type;
        if (type == FF_JNI_CLASS) {
            jclass clazz;
            last_clazz = NULL;
            clazz = (*env)->FindClass(env, jfields_mapping[i].name);
            if ((ret = ff_jni_exception_check(env, mandatory, log_ctx)) < 0 && mandatory) {
                goto done;
            last_clazz = *(jclass*)((uint8_t*)jfields + jfields_mapping[i].offset) =
                    global ? (*env)->NewGlobalRef(env, clazz) : clazz;
        } else {
            if (!last_clazz) {
                ret = AVERROR_EXTERNAL;
                break;
            switch(type) {
            case FF_JNI_FIELD: {
                jfieldID field_id = (*env)->GetFieldID(env, last_clazz, jfields_mapping[i].method, jfields_mapping[i].signature);
                if ((ret = ff_jni_exception_check(env, mandatory, log_ctx)) < 0 && mandatory) {
                    goto done;
                *(jfieldID*)((uint8_t*)jfields + jfields_mapping[i].offset) = field_id;
                break;
            case FF_JNI_STATIC_FIELD: {
                jfieldID field_id = (*env)->GetStaticFieldID(env, last_clazz, jfields_mapping[i].method, jfields_mapping[i].signature);
                if ((ret = ff_jni_exception_check(env, mandatory, log_ctx)) < 0 && mandatory) {
                    goto done;
                *(jfieldID*)((uint8_t*)jfields + jfields_mapping[i].offset) = field_id;
                break;
            case FF_JNI_METHOD: {
                jmethodID method_id = (*env)->GetMethodID(env, last_clazz, jfields_mapping[i].method, jfields_mapping[i].signature);
                if ((ret = ff_jni_exception_check(env, mandatory, log_ctx)) < 0 && mandatory) {
                    goto done;
                *(jmethodID*)((uint8_t*)jfields + jfields_mapping[i].offset) = method_id;
                break;
            case FF_JNI_STATIC_METHOD: {
                jmethodID method_id = (*env)->GetStaticMethodID(env, last_clazz, jfields_mapping[i].method, jfields_mapping[i].signature);
                if ((ret = ff_jni_exception_check(env, mandatory, log_ctx)) < 0 && mandatory) {
                    goto done;
                *(jmethodID*)((uint8_t*)jfields + jfields_mapping[i].offset) = method_id;
                break;
            default:
                av_log(log_ctx, AV_LOG_ERROR, "Unknown JNI field type\n");
                ret = AVERROR(EINVAL);
                goto done;
            ret = 0;
done:
    if (ret < 0) {
        /* reset jfields in case of failure so it does not leak references */
        ff_jni_reset_jfields(env, jfields, jfields_mapping, global, log_ctx);
    return ret;