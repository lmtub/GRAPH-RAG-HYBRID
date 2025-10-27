char *ff_AMediaCodecList_getCodecNameByType(const char *mime, int width, int height, void *log_ctx)

{

    int ret;

    char *name = NULL;

    char *supported_type = NULL;



    int attached = 0;

    JNIEnv *env = NULL;

    struct JNIAMediaCodecListFields jfields = { 0 };



    jobject format = NULL;

    jobject codec = NULL;

    jstring tmp = NULL;



    jobject info = NULL;

    jobject type = NULL;

    jobjectArray types = NULL;



    JNI_ATTACH_ENV_OR_RETURN(env, &attached, log_ctx, NULL);



    if ((ret = ff_jni_init_jfields(env, &jfields, jni_amediacodeclist_mapping, 0, log_ctx)) < 0) {

        goto done;

    }



    if (jfields.init_id && jfields.find_decoder_for_format_id) {

        tmp = ff_jni_utf_chars_to_jstring(env, mime, log_ctx);

        if (!tmp) {

            goto done;

        }



        format = (*env)->CallStaticObjectMethod(env, jfields.mediaformat_class, jfields.create_video_format_id, tmp, width, height);

        if (ff_jni_exception_check(env, 1, log_ctx) < 0) {

            goto done;

        }

        (*env)->DeleteLocalRef(env, tmp);

        tmp = NULL;



        codec = (*env)->NewObject(env, jfields.mediacodec_list_class, jfields.init_id, 0);

        if (ff_jni_exception_check(env, 1, log_ctx) < 0) {

            goto done;

        }



        tmp = (*env)->CallObjectMethod(env, codec, jfields.find_decoder_for_format_id, format);

        if (ff_jni_exception_check(env, 1, log_ctx) < 0) {

            goto done;

        }

        if (!tmp) {

            av_log(NULL, AV_LOG_ERROR, "Could not find decoder in media codec list "

                                       "for format { mime=%s width=%d height=%d }\n", mime, width, height);

            goto done;

        }



        name = ff_jni_jstring_to_utf_chars(env, tmp, log_ctx);

        if (!name) {

            goto done;

        }



    } else {

        int i;

        int codec_count;



        codec_count = (*env)->CallStaticIntMethod(env, jfields.mediacodec_list_class, jfields.get_codec_count_id);

        if (ff_jni_exception_check(env, 1, log_ctx) < 0) {

            goto done;

        }



        for(i = 0; i < codec_count; i++) {

            int j;

            int type_count;

            int is_encoder;



            info = (*env)->CallStaticObjectMethod(env, jfields.mediacodec_list_class, jfields.get_codec_info_at_id, i);

            if (ff_jni_exception_check(env, 1, log_ctx) < 0) {

                goto done;

            }



            types = (*env)->CallObjectMethod(env, info, jfields.get_supported_types_id);

            if (ff_jni_exception_check(env, 1, log_ctx) < 0) {

                goto done;

            }



            is_encoder = (*env)->CallBooleanMethod(env, info, jfields.is_encoder_id);

            if (ff_jni_exception_check(env, 1, log_ctx) < 0) {

                goto done;

            }



            if (is_encoder) {

                continue;

            }



            type_count = (*env)->GetArrayLength(env, types);

            for (j = 0; j < type_count; j++) {



                type = (*env)->GetObjectArrayElement(env, types, j);

                if (ff_jni_exception_check(env, 1, log_ctx) < 0) {

                    goto done;

                }



                supported_type = ff_jni_jstring_to_utf_chars(env, type, log_ctx);

                if (!supported_type) {

                    goto done;

                }



                if (!av_strcasecmp(supported_type, mime)) {

                    jobject codec_name;



                    codec_name = (*env)->CallObjectMethod(env, info, jfields.get_name_id);

                    if (ff_jni_exception_check(env, 1, log_ctx) < 0) {

                        goto done;

                    }



                    name = ff_jni_jstring_to_utf_chars(env, codec_name, log_ctx);

                    if (!name) {

                        goto done;

                    }



                    if (strstr(name, "OMX.google")) {

                        av_freep(&name);

                        continue;

                    }

                }



                av_freep(&supported_type);

            }



            (*env)->DeleteLocalRef(env, info);

            info = NULL;



            (*env)->DeleteLocalRef(env, types);

            types = NULL;



            if (name)

                break;

        }

    }



done:

    if (format) {

        (*env)->DeleteLocalRef(env, format);

    }



    if (codec) {

        (*env)->DeleteLocalRef(env, codec);

    }



    if (tmp) {

        (*env)->DeleteLocalRef(env, tmp);

    }



    if (info) {

        (*env)->DeleteLocalRef(env, info);

    }



    if (type) {

        (*env)->DeleteLocalRef(env, type);

    }



    if (types) {

        (*env)->DeleteLocalRef(env, types);

    }



    av_freep(&supported_type);



    ff_jni_reset_jfields(env, &jfields, jni_amediacodeclist_mapping, 0, log_ctx);



    JNI_DETACH_ENV(attached, log_ctx);



    return name;

}
