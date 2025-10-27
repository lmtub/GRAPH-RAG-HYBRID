FFAMediaCodec* ff_AMediaCodec_createDecoderByType(const char *mime)

{

    JNIEnv *env = NULL;

    FFAMediaCodec *codec = NULL;

    jstring mime_type = NULL;



    codec = av_mallocz(sizeof(FFAMediaCodec));

    if (!codec) {

        return NULL;

    }

    codec->class = &amediacodec_class;



    env = ff_jni_get_env(codec);

    if (!env) {

        av_freep(&codec);

        return NULL;

    }



    if (ff_jni_init_jfields(env, &codec->jfields, jni_amediacodec_mapping, 1, codec) < 0) {

        goto fail;

    }



    mime_type = ff_jni_utf_chars_to_jstring(env, mime, codec);

    if (!mime_type) {

        goto fail;

    }



    codec->object = (*env)->CallStaticObjectMethod(env, codec->jfields.mediacodec_class, codec->jfields.create_decoder_by_type_id, mime_type);

    if (ff_jni_exception_check(env, 1, codec) < 0) {

        goto fail;

    }



    codec->object = (*env)->NewGlobalRef(env, codec->object);

    if (!codec->object) {

        goto fail;

    }



    if (codec_init_static_fields(codec) < 0) {

        goto fail;

    }



    if (codec->jfields.get_input_buffer_id && codec->jfields.get_output_buffer_id) {

        codec->has_get_i_o_buffer = 1;

    }



    return codec;

fail:

    ff_jni_reset_jfields(env, &codec->jfields, jni_amediacodec_mapping, 1, codec);



    if (mime_type) {

        (*env)->DeleteLocalRef(env, mime_type);

    }



    av_freep(&codec);



    return NULL;

}
