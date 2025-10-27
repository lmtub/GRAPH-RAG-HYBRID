static int theora_header(AVFormatContext *s, int idx)
{
    struct ogg *ogg       = s->priv_data;
    struct ogg_stream *os = ogg->streams + idx;
    AVStream *st          = s->streams[idx];
    TheoraParams *thp     = os->private;
    int cds               = st->codec->extradata_size + os->psize + 2;
    int err;
    uint8_t *cdp;
    if (!(os->buf[os->pstart] & 0x80))
        return 0;
    if (!thp) {
        thp = av_mallocz(sizeof(*thp));
        if (!thp)
            return AVERROR(ENOMEM);
        os->private = thp;
    }
    switch (os->buf[os->pstart]) {
    case 0x80: {
        GetBitContext gb;
        AVRational timebase;
        init_get_bits(&gb, os->buf + os->pstart, os->psize * 8);
        /* 0x80"theora" */
        skip_bits_long(&gb, 7 * 8);
        thp->version = get_bits_long(&gb, 24);
        if (thp->version < 0x030100) {
            av_log(s, AV_LOG_ERROR,
                   "Too old or unsupported Theora (%x)\n", thp->version);
            return AVERROR(ENOSYS);
        }
        st->codec->width  = get_bits(&gb, 16) << 4;
        st->codec->height = get_bits(&gb, 16) << 4;
        if (thp->version >= 0x030400)
            skip_bits(&gb, 100);
        if (thp->version >= 0x030200) {
            int width  = get_bits_long(&gb, 24);
            int height = get_bits_long(&gb, 24);
            if (width  <= st->codec->width  && width  > st->codec->width  - 16 &&
                height <= st->codec->height && height > st->codec->height - 16) {
                st->codec->width  = width;
                st->codec->height = height;
            }
            skip_bits(&gb, 16);
        }
        timebase.den = get_bits_long(&gb, 32);
        timebase.num = get_bits_long(&gb, 32);
        if (!(timebase.num > 0 && timebase.den > 0)) {
            av_log(s, AV_LOG_WARNING, "Invalid time base in theora stream, assuming 25 FPS\n");
            timebase.num = 1;
            timebase.den = 25;
        }
        avpriv_set_pts_info(st, 64, timebase.num, timebase.den);
        st->sample_aspect_ratio.num = get_bits_long(&gb, 24);
        st->sample_aspect_ratio.den = get_bits_long(&gb, 24);
        if (thp->version >= 0x030200)
            skip_bits_long(&gb, 38);
        if (thp->version >= 0x304000)
            skip_bits(&gb, 2);
        thp->gpshift = get_bits(&gb, 5);
        thp->gpmask  = (1 << thp->gpshift) - 1;
        st->codec->codec_type = AVMEDIA_TYPE_VIDEO;
        st->codec->codec_id   = AV_CODEC_ID_THEORA;
        st->need_parsing      = AVSTREAM_PARSE_HEADERS;
    }
    break;
    case 0x81:
        ff_vorbis_comment(s, &st->metadata, os->buf + os->pstart + 7, os->psize - 7);
    case 0x82:
        if (!thp->version)
            return AVERROR_INVALIDDATA;
        break;
    default:
        av_log(s, AV_LOG_ERROR, "Unknown header type %X\n", os->buf[os->pstart]);
        return AVERROR_INVALIDDATA;
    }
    if ((err = av_reallocp(&st->codec->extradata,
                           cds + FF_INPUT_BUFFER_PADDING_SIZE)) < 0) {
        st->codec->extradata_size = 0;
        return err;
    }
    cdp    = st->codec->extradata + st->codec->extradata_size;
    *cdp++ = os->psize >> 8;
    *cdp++ = os->psize & 0xff;
    memcpy(cdp, os->buf + os->pstart, os->psize);
    st->codec->extradata_size = cds;
    return 1;
}