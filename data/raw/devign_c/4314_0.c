static enum CodecID find_codec_or_die(const char *name, enum AVMediaType type, int encoder)

{

    const char *codec_string = encoder ? "encoder" : "decoder";

    AVCodec *codec;



    if(!name)

        return CODEC_ID_NONE;

    codec = encoder ?

        avcodec_find_encoder_by_name(name) :

        avcodec_find_decoder_by_name(name);

    if(!codec) {

        av_log(NULL, AV_LOG_FATAL, "Unknown %s '%s'\n", codec_string, name);

        exit_program(1);

    }

    if(codec->type != type) {

        av_log(NULL, AV_LOG_FATAL, "Invalid %s type '%s'\n", codec_string, name);

        exit_program(1);

    }

    return codec->id;

}
