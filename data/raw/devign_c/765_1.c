theora_header (AVFormatContext * s, int idx)

{

    struct ogg *ogg = s->priv_data;

    struct ogg_stream *os = ogg->streams + idx;

    AVStream *st = s->streams[idx];

    struct theora_params *thp = os->private;

    int cds = st->codec->extradata_size + os->psize + 2;

    uint8_t *cdp;



    if(!(os->buf[os->pstart] & 0x80))

        return 0;



    if(!thp){

        thp = av_mallocz(sizeof(*thp));

        os->private = thp;




    if (os->buf[os->pstart] == 0x80) {

        GetBitContext gb;

        int width, height;



        init_get_bits(&gb, os->buf + os->pstart, os->psize*8);



        skip_bits_long(&gb, 7*8); /* 0x80"theora" */



        thp->version = get_bits_long(&gb, 24);

        if (thp->version < 0x030100)

        {

            av_log(s, AV_LOG_ERROR,

                "Too old or unsupported Theora (%x)\n", thp->version);

            return -1;




        width  = get_bits(&gb, 16) << 4;

        height = get_bits(&gb, 16) << 4;

        avcodec_set_dimensions(st->codec, width, height);



        if (thp->version >= 0x030400)

            skip_bits(&gb, 100);



        if (thp->version >= 0x030200) {

            width  = get_bits_long(&gb, 24);

            height = get_bits_long(&gb, 24);

            if (   width  <= st->codec->width  && width  > st->codec->width-16

                && height <= st->codec->height && height > st->codec->height-16)

                avcodec_set_dimensions(st->codec, width, height);



            skip_bits(&gb, 16);


        st->codec->time_base.den = get_bits_long(&gb, 32);

        st->codec->time_base.num = get_bits_long(&gb, 32);






        st->time_base = st->codec->time_base;



        st->sample_aspect_ratio.num = get_bits_long(&gb, 24);

        st->sample_aspect_ratio.den = get_bits_long(&gb, 24);



        if (thp->version >= 0x030200)

            skip_bits_long(&gb, 38);

        if (thp->version >= 0x304000)

            skip_bits(&gb, 2);



        thp->gpshift = get_bits(&gb, 5);

        thp->gpmask = (1 << thp->gpshift) - 1;



        st->codec->codec_type = CODEC_TYPE_VIDEO;

        st->codec->codec_id = CODEC_ID_THEORA;



    } else if (os->buf[os->pstart] == 0x83) {

        vorbis_comment (s, os->buf + os->pstart + 7, os->psize - 8);




    st->codec->extradata = av_realloc (st->codec->extradata,

                                       cds + FF_INPUT_BUFFER_PADDING_SIZE);

    cdp = st->codec->extradata + st->codec->extradata_size;

    *cdp++ = os->psize >> 8;

    *cdp++ = os->psize & 0xff;

    memcpy (cdp, os->buf + os->pstart, os->psize);

    st->codec->extradata_size = cds;



    return 1;
