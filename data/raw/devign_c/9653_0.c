static AVCodec *choose_codec(OptionsContext *o, AVFormatContext *s, AVStream *st, enum AVMediaType type)

{

    char *codec_name = NULL;



    MATCH_PER_STREAM_OPT(codec_names, str, codec_name, s, st);



    if (!codec_name) {

        if (s->oformat) {

            st->codec->codec_id = av_guess_codec(s->oformat, NULL, s->filename, NULL, type);

            return avcodec_find_encoder(st->codec->codec_id);

        }

    } else if (!strcmp(codec_name, "copy"))

        st->stream_copy = 1;

    else {

        st->codec->codec_id = find_codec_or_die(codec_name, type, s->iformat == NULL);

        return s->oformat ? avcodec_find_encoder_by_name(codec_name) :

                            avcodec_find_decoder_by_name(codec_name);

    }



    return NULL;

}
