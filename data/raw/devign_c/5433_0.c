static int mpegps_read_header(AVFormatContext *s)

{

    MpegDemuxContext *m = s->priv_data;

    char buffer[7];

    int64_t last_pos = avio_tell(s->pb);



    m->header_state = 0xff;

    s->ctx_flags   |= AVFMTCTX_NOHEADER;



    avio_get_str(s->pb, 6, buffer, sizeof(buffer));

    if (!memcmp("IMKH", buffer, 4)) {

        m->imkh_cctv = 1;

    } else if (!memcmp("Sofdec", buffer, 6)) {

        m->sofdec = 1;

    } else

       avio_seek(s->pb, last_pos, SEEK_SET);



    /* no need to do more */

    return 0;

}
