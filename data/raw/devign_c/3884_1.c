static int opus_header(AVFormatContext *avf, int idx)

{

    struct ogg *ogg              = avf->priv_data;

    struct ogg_stream *os        = &ogg->streams[idx];

    AVStream *st                 = avf->streams[idx];

    struct oggopus_private *priv = os->private;

    uint8_t *packet              = os->buf + os->pstart;



    if (!priv) {

        priv = os->private = av_mallocz(sizeof(*priv));

        if (!priv)

            return AVERROR(ENOMEM);

    }



    if (os->flags & OGG_FLAG_BOS) {

        if (os->psize < OPUS_HEAD_SIZE || (AV_RL8(packet + 8) & 0xF0) != 0)

            return AVERROR_INVALIDDATA;

        st->codecpar->codec_type = AVMEDIA_TYPE_AUDIO;

        st->codecpar->codec_id   = AV_CODEC_ID_OPUS;

        st->codecpar->channels   = AV_RL8(packet + 9);



        priv->pre_skip        = AV_RL16(packet + 10);

        st->codecpar->initial_padding = priv->pre_skip;

        /*orig_sample_rate    = AV_RL32(packet + 12);*/

        /*gain                = AV_RL16(packet + 16);*/

        /*channel_map         = AV_RL8 (packet + 18);*/




        if (ff_alloc_extradata(st->codecpar, os->psize))

            return AVERROR(ENOMEM);



        memcpy(st->codecpar->extradata, packet, os->psize);



        st->codecpar->sample_rate = 48000;

        st->codecpar->seek_preroll = av_rescale(OPUS_SEEK_PREROLL_MS,

                                                st->codecpar->sample_rate, 1000);

        avpriv_set_pts_info(st, 64, 1, 48000);

        priv->need_comments = 1;

        return 1;

    }



    if (priv->need_comments) {

        if (os->psize < 8 || memcmp(packet, "OpusTags", 8))

            return AVERROR_INVALIDDATA;

        ff_vorbis_stream_comment(avf, st, packet + 8, os->psize - 8);

        priv->need_comments--;

        return 1;

    }



    return 0;

}