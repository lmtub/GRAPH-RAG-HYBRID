int av_write_header(AVFormatContext *s)

{

    int ret, i;

    AVStream *st;



    // some sanity checks

    if (s->nb_streams == 0) {

        av_log(s, AV_LOG_ERROR, "no streams\n");

        return AVERROR(EINVAL);

    }



    for(i=0;i<s->nb_streams;i++) {

        st = s->streams[i];



        switch (st->codec->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            if(st->codec->sample_rate<=0){

                av_log(s, AV_LOG_ERROR, "sample rate not set\n");

                return AVERROR(EINVAL);

            }

            if(!st->codec->block_align)

                st->codec->block_align = st->codec->channels *

                    av_get_bits_per_sample(st->codec->codec_id) >> 3;

            break;

        case AVMEDIA_TYPE_VIDEO:

            if(st->codec->time_base.num<=0 || st->codec->time_base.den<=0){ //FIXME audio too?

                av_log(s, AV_LOG_ERROR, "time base not set\n");

                return AVERROR(EINVAL);

            }

            if((st->codec->width<=0 || st->codec->height<=0) && !(s->oformat->flags & AVFMT_NODIMENSIONS)){

                av_log(s, AV_LOG_ERROR, "dimensions not set\n");

                return AVERROR(EINVAL);

            }

            if(av_cmp_q(st->sample_aspect_ratio, st->codec->sample_aspect_ratio)){

                av_log(s, AV_LOG_ERROR, "Aspect ratio mismatch between encoder and muxer layer\n");

                return AVERROR(EINVAL);

            }

            break;

        }



        if(s->oformat->codec_tag){

            if(st->codec->codec_tag && st->codec->codec_id == CODEC_ID_RAWVIDEO && av_codec_get_tag(s->oformat->codec_tag, st->codec->codec_id) == 0 && !validate_codec_tag(s, st)){

                //the current rawvideo encoding system ends up setting the wrong codec_tag for avi, we override it here

                st->codec->codec_tag= 0;

            }

            if(st->codec->codec_tag){

                if (!validate_codec_tag(s, st)) {

                    char tagbuf[32];

                    av_get_codec_tag_string(tagbuf, sizeof(tagbuf), st->codec->codec_tag);

                    av_log(s, AV_LOG_ERROR,

                           "Tag %s/0x%08x incompatible with output codec '%s'\n",

                           tagbuf, st->codec->codec_tag, st->codec->codec->name);

                    return AVERROR_INVALIDDATA;

                }

            }else

                st->codec->codec_tag= av_codec_get_tag(s->oformat->codec_tag, st->codec->codec_id);

        }



        if(s->oformat->flags & AVFMT_GLOBALHEADER &&

            !(st->codec->flags & CODEC_FLAG_GLOBAL_HEADER))

          av_log(s, AV_LOG_WARNING, "Codec for stream %d does not use global headers but container format requires global headers\n", i);

    }



    if (!s->priv_data && s->oformat->priv_data_size > 0) {

        s->priv_data = av_mallocz(s->oformat->priv_data_size);

        if (!s->priv_data)

            return AVERROR(ENOMEM);

    }



#if LIBAVFORMAT_VERSION_MAJOR < 53

    ff_metadata_mux_compat(s);

#endif



    /* set muxer identification string */

    if (!(s->streams[0]->codec->flags & CODEC_FLAG_BITEXACT)) {

        AVMetadata *m;

        AVMetadataTag *t;



        if (!(m = av_mallocz(sizeof(AVMetadata))))

            return AVERROR(ENOMEM);

        av_metadata_set2(&m, "encoder", LIBAVFORMAT_IDENT, 0);

        metadata_conv(&m, s->oformat->metadata_conv, NULL);

        if ((t = av_metadata_get(m, "", NULL, AV_METADATA_IGNORE_SUFFIX)))

            av_metadata_set2(&s->metadata, t->key, t->value, 0);

        av_metadata_free(&m);

    }



    if(s->oformat->write_header){

        ret = s->oformat->write_header(s);

        if (ret < 0)

            return ret;

    }



    /* init PTS generation */

    for(i=0;i<s->nb_streams;i++) {

        int64_t den = AV_NOPTS_VALUE;

        st = s->streams[i];



        switch (st->codec->codec_type) {

        case AVMEDIA_TYPE_AUDIO:

            den = (int64_t)st->time_base.num * st->codec->sample_rate;

            break;

        case AVMEDIA_TYPE_VIDEO:

            den = (int64_t)st->time_base.num * st->codec->time_base.den;

            break;

        default:

            break;

        }

        if (den != AV_NOPTS_VALUE) {

            if (den <= 0)

                return AVERROR_INVALIDDATA;

            av_frac_init(&st->pts, 0, 0, den);

        }

    }

    return 0;

}
