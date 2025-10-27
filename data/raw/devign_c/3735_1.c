static int aiff_read_packet(AVFormatContext *s,

                            AVPacket *pkt)

{

    AVStream *st = s->streams[0];

    AIFFInputContext *aiff = s->priv_data;

    int64_t max_size;

    int res, size;



    /* calculate size of remaining data */

    max_size = aiff->data_end - avio_tell(s->pb);

    if (max_size <= 0)

        return AVERROR_EOF;



    /* Now for that packet */

    if (st->codec->block_align >= 33) // GSM, QCLP, IMA4

        size = st->codec->block_align;

    else

        size = (MAX_SIZE / st->codec->block_align) * st->codec->block_align;

    size = FFMIN(max_size, size);

    res = av_get_packet(s->pb, pkt, size);

    if (res < 0)

        return res;





    /* Only one stream in an AIFF file */

    pkt->stream_index = 0;

    pkt->duration     = (res / st->codec->block_align) * aiff->block_duration;

    return 0;

}