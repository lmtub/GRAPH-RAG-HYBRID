static enum CodecID find_codec_or_die(const char *name, int type, int encoder, int strict)

{

    const char *codec_string = encoder ? "encoder" : "decoder";

    AVCodec *codec;



    if(!name)

        return CODEC_ID_NONE;

    codec = encoder ?

        avcodec_find_encoder_by_name(name) :

        avcodec_find_decoder_by_name(name);

    if(!codec) {

        fprintf(stderr, "Unknown %s '%s'\n", codec_string, name);

        ffmpeg_exit(1);

    }

    if(codec->type != type) {

        fprintf(stderr, "Invalid %s type '%s'\n", codec_string, name);

        ffmpeg_exit(1);

    }

    if(codec->capabilities & CODEC_CAP_EXPERIMENTAL &&

       strict > FF_COMPLIANCE_EXPERIMENTAL) {

        fprintf(stderr, "%s '%s' is experimental and might produce bad "

                "results.\nAdd '-strict experimental' if you want to use it.\n",

                codec_string, codec->name);

        codec = encoder ?

            avcodec_find_encoder(codec->id) :

            avcodec_find_decoder(codec->id);

        if (!(codec->capabilities & CODEC_CAP_EXPERIMENTAL))

            fprintf(stderr, "Or use the non experimental %s '%s'.\n",

                    codec_string, codec->name);

        ffmpeg_exit(1);

    }

    return codec->id;

}
