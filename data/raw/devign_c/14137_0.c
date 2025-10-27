static int wv_read_header(AVFormatContext *s)

{

    AVIOContext *pb = s->pb;

    WVContext *wc = s->priv_data;

    AVStream *st;

    int ret;



    wc->block_parsed = 0;

    for (;;) {

        if ((ret = wv_read_block_header(s, pb, 0)) < 0)

            return ret;

        if (!AV_RN32(wc->extra))

            avio_skip(pb, wc->blksize - 24);

        else

            break;

    }



    /* now we are ready: build format streams */

    st = avformat_new_stream(s, NULL);

    if (!st)

        return AVERROR(ENOMEM);

    st->codec->codec_type            = AVMEDIA_TYPE_AUDIO;

    st->codec->codec_id              = AV_CODEC_ID_WAVPACK;

    st->codec->channels              = wc->chan;

    st->codec->channel_layout        = wc->chmask;

    st->codec->sample_rate           = wc->rate;

    st->codec->bits_per_coded_sample = wc->bpp;

    avpriv_set_pts_info(st, 64, 1, wc->rate);

    st->start_time = 0;

    st->duration   = wc->samples;



    if (s->pb->seekable) {

        int64_t cur = avio_tell(s->pb);

        wc->apetag_start = ff_ape_parse_tag(s);

        if (!av_dict_get(s->metadata, "", NULL, AV_DICT_IGNORE_SUFFIX))

            ff_id3v1_read(s);

        avio_seek(s->pb, cur, SEEK_SET);

    }



    return 0;

}
