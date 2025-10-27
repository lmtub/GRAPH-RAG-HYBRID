static int mpegps_read_packet(AVFormatContext *s,

                              AVPacket *pkt)

{

    AVStream *st;

    int len, startcode, i, type, codec_id;

    int64_t pts, dts;



 redo:

    len = mpegps_read_pes_header(s, NULL, &startcode, &pts, &dts, 1);

    if (len < 0)

        return len;

    

    /* now find stream */

    for(i=0;i<s->nb_streams;i++) {

        st = s->streams[i];

        if (st->id == startcode)

            goto found;

    }

    if (startcode >= 0x1e0 && startcode <= 0x1ef) {

        type = CODEC_TYPE_VIDEO;

        codec_id = CODEC_ID_MPEG1VIDEO;

    } else if (startcode >= 0x1c0 && startcode <= 0x1df) {

        type = CODEC_TYPE_AUDIO;

        codec_id = CODEC_ID_MP2;

    } else if (startcode >= 0x80 && startcode <= 0x9f) {

        type = CODEC_TYPE_AUDIO;

        codec_id = CODEC_ID_AC3;

    } else if (startcode >= 0xa0 && startcode <= 0xbf) {

        type = CODEC_TYPE_AUDIO;

        codec_id = CODEC_ID_PCM_S16BE;

    } else {

    skip:

        /* skip packet */

        url_fskip(&s->pb, len);

        goto redo;

    }

    /* no stream found: add a new stream */

    st = av_new_stream(s, startcode);

    if (!st) 

        goto skip;

    st->codec.codec_type = type;

    st->codec.codec_id = codec_id;

    if (codec_id != CODEC_ID_PCM_S16BE)

        st->need_parsing = 1;

 found:

    if (startcode >= 0xa0 && startcode <= 0xbf) {

        int b1, freq;

        static const int lpcm_freq_tab[4] = { 48000, 96000, 44100, 32000 };



        /* for LPCM, we just skip the header and consider it is raw

           audio data */

        if (len <= 3)

            goto skip;

        get_byte(&s->pb); /* emphasis (1), muse(1), reserved(1), frame number(5) */

        b1 = get_byte(&s->pb); /* quant (2), freq(2), reserved(1), channels(3) */

        get_byte(&s->pb); /* dynamic range control (0x80 = off) */

        len -= 3;

        freq = (b1 >> 4) & 3;

        st->codec.sample_rate = lpcm_freq_tab[freq];

        st->codec.channels = 1 + (b1 & 7);

        st->codec.bit_rate = st->codec.channels * st->codec.sample_rate * 2;

    }

    av_new_packet(pkt, len);

    get_buffer(&s->pb, pkt->data, pkt->size);

    pkt->pts = pts;

    pkt->dts = dts;

    pkt->stream_index = st->index;

#if 0

    printf("%d: pts=%0.3f dts=%0.3f\n",

           pkt->stream_index, pkt->pts / 90000.0, pkt->dts / 90000.0);

#endif

    return 0;

}
