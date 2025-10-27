static void mpegts_insert_pcr_only(AVFormatContext *s, AVStream *st)

{

    MpegTSWrite *ts = s->priv_data;

    MpegTSWriteStream *ts_st = st->priv_data;

    uint8_t *q;

    uint8_t buf[TS_PACKET_SIZE];



    q    = buf;

    *q++ = 0x47;

    *q++ = ts_st->pid >> 8;

    *q++ = ts_st->pid;

    *q++ = 0x20 | ts_st->cc;   /* Adaptation only */

    /* Continuity Count field does not increment (see 13818-1 section 2.4.3.3) */

    *q++ = TS_PACKET_SIZE - 5; /* Adaptation Field Length */

    *q++ = 0x10;               /* Adaptation flags: PCR present */







    /* PCR coded into 6 bytes */

    q += write_pcr_bits(q, get_pcr(ts, s->pb));



    /* stuffing bytes */

    memset(q, 0xFF, TS_PACKET_SIZE - (q - buf));

    mpegts_prefix_m2ts_header(s);

    avio_write(s->pb, buf, TS_PACKET_SIZE);
