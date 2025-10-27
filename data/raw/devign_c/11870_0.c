static int mp3_read_header(AVFormatContext *s,

                           AVFormatParameters *ap)

{

    AVStream *st;

    int64_t off;



    st = av_new_stream(s, 0);

    if (!st)

        return AVERROR(ENOMEM);



    st->codec->codec_type = CODEC_TYPE_AUDIO;

    st->codec->codec_id = CODEC_ID_MP3;

    st->need_parsing = AVSTREAM_PARSE_FULL;

    st->start_time = 0;



    ff_id3v1_read(s);

    ff_id3v2_read(s);



    off = url_ftell(s->pb);

    if (mp3_parse_vbr_tags(s, st, off) < 0)

        url_fseek(s->pb, off, SEEK_SET);



    /* the parameters will be extracted from the compressed bitstream */

    return 0;

}
