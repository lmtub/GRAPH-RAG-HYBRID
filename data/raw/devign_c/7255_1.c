static int dirac_header(AVFormatContext *s, int idx)

{

    struct ogg *ogg = s->priv_data;

    struct ogg_stream *os = ogg->streams + idx;

    AVStream *st = s->streams[idx];

    dirac_source_params source;

    GetBitContext gb;



    // already parsed the header

    if (st->codec->codec_id == AV_CODEC_ID_DIRAC)

        return 0;



    init_get_bits(&gb, os->buf + os->pstart + 13, (os->psize - 13) * 8);

    if (avpriv_dirac_parse_sequence_header(st->codec, &gb, &source) < 0)

        return -1;



    st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

    st->codec->codec_id = AV_CODEC_ID_DIRAC;

    // dirac in ogg always stores timestamps as though the video were interlaced

    avpriv_set_pts_info(st, 64, st->codec->framerate.den, 2*st->codec->framerate.num);

    return 1;

}
