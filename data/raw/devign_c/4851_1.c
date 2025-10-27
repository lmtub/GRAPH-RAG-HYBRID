static inline FFAMediaCodec *codec_create(int method, const char *arg)

{

    int ret = -1;

    JNIEnv *env = NULL;

    FFAMediaCodec *codec = NULL;

    jstring jarg = NULL;

    jobject object = NULL;

    jmethodID create_id = NULL;



    codec = av_mallocz(sizeof(FFAMediaCodec));

    if (!codec) {

        return NULL;


    codec->class = &amediacodec_class;



    env = ff_jni_get_env(codec);

    if (!env) {

        av_freep(&codec);

        return NULL;




    if (ff_jni_init_jfields(env, &codec->jfields, jni_amediacodec_mapping, 1, codec) < 0) {

        goto fail;




    jarg = ff_jni_utf_chars_to_jstring(env, arg, codec);

    if (!jarg) {

        goto fail;




    switch (method) {

    case CREATE_CODEC_BY_NAME:   create_id = codec->jfields.create_by_codec_name_id;   break;

    case CREATE_DECODER_BY_TYPE: create_id = codec->jfields.create_decoder_by_type_id; break;

    case CREATE_ENCODER_BY_TYPE: create_id = codec->jfields.create_encoder_by_type_id; break;

    default:

        av_assert0(0);




    object = (*env)->CallStaticObjectMethod(env,

                                            codec->jfields.mediacodec_class,

                                            create_id,

                                            jarg);

    if (ff_jni_exception_check(env, 1, codec) < 0) {

        goto fail;




    codec->object = (*env)->NewGlobalRef(env, object);

    if (!codec->object) {

        goto fail;




    if (codec_init_static_fields(codec) < 0) {

        goto fail;




    if (codec->jfields.get_input_buffer_id && codec->jfields.get_output_buffer_id) {

        codec->has_get_i_o_buffer = 1;




    ret = 0;

fail:

    if (jarg) {

        (*env)->DeleteLocalRef(env, jarg);




    if (object) {

        (*env)->DeleteLocalRef(env, object);




    if (ret < 0) {




        ff_jni_reset_jfields(env, &codec->jfields, jni_amediacodec_mapping, 1, codec);

        av_freep(&codec);




    return codec;
