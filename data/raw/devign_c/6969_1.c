static int txd_read_header(AVFormatContext *s, AVFormatParameters *ap) {
    AVStream *st;
    st = avformat_new_stream(s, NULL);
    if (!st)
        return AVERROR(ENOMEM);
    st->codec->codec_type = AVMEDIA_TYPE_VIDEO;
    st->codec->codec_id = CODEC_ID_TXD;
    st->codec->time_base.den = 5;
    st->codec->time_base.num = 1;
    /* the parameters will be extracted from the compressed bitstream */
    return 0;
}