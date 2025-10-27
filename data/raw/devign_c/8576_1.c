static int speex_header(AVFormatContext *s, int idx) {

    struct ogg *ogg = s->priv_data;

    struct ogg_stream *os = ogg->streams + idx;

    struct speex_params *spxp = os->private;

    AVStream *st = s->streams[idx];

    uint8_t *p = os->buf + os->pstart;



    if (!spxp) {

        spxp = av_mallocz(sizeof(*spxp));

        os->private = spxp;

    }



    if (spxp->seq > 1)

        return 0;



    if (spxp->seq == 0) {

        int frames_per_packet;

        st->codec->codec_type = AVMEDIA_TYPE_AUDIO;

        st->codec->codec_id = AV_CODEC_ID_SPEEX;



        if (os->psize < 68) {

            av_log(s, AV_LOG_ERROR, "speex packet too small\n");

            return AVERROR_INVALIDDATA;

        }



        st->codec->sample_rate = AV_RL32(p + 36);

        st->codec->channels = AV_RL32(p + 48);

        if (st->codec->channels < 1 || st->codec->channels > 2) {

            av_log(s, AV_LOG_ERROR, "invalid channel count. Speex must be mono or stereo.\n");

            return AVERROR_INVALIDDATA;

        }

        st->codec->channel_layout = st->codec->channels == 1 ? AV_CH_LAYOUT_MONO :

                                                               AV_CH_LAYOUT_STEREO;



        spxp->packet_size  = AV_RL32(p + 56);

        frames_per_packet  = AV_RL32(p + 64);

        if (frames_per_packet)

            spxp->packet_size *= frames_per_packet;



        ff_alloc_extradata(st->codec, os->psize);

        memcpy(st->codec->extradata, p, st->codec->extradata_size);



        avpriv_set_pts_info(st, 64, 1, st->codec->sample_rate);

    } else

        ff_vorbis_comment(s, &st->metadata, p, os->psize);



    spxp->seq++;

    return 1;

}
