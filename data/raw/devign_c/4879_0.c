static int mp3_parse_vbr_tags(AVFormatContext *s, AVStream *st, int64_t base)

{

    uint32_t v, spf;

    MPADecodeHeader c;

    int vbrtag_size = 0;

    MP3DecContext *mp3 = s->priv_data;



    ffio_init_checksum(s->pb, ff_crcA001_update, 0);



    v = avio_rb32(s->pb);

    if(ff_mpa_check_header(v) < 0)

      return -1;



    if (avpriv_mpegaudio_decode_header(&c, v) == 0)

        vbrtag_size = c.frame_size;

    if(c.layer != 3)

        return -1;



    spf = c.lsf ? 576 : 1152; /* Samples per frame, layer 3 */



    mp3->frames = 0;

    mp3->size   = 0;



    mp3_parse_info_tag(s, st, &c, spf);

    mp3_parse_vbri_tag(s, st, base);



    if (!mp3->frames && !mp3->size)

        return -1;



    /* Skip the vbr tag frame */

    avio_seek(s->pb, base + vbrtag_size, SEEK_SET);



    if (mp3->frames)

        st->duration = av_rescale_q(mp3->frames, (AVRational){spf, c.sample_rate},

                                    st->time_base);

    if (mp3->size && mp3->frames && !mp3->is_cbr)

        st->codec->bit_rate = av_rescale(mp3->size, 8 * c.sample_rate, mp3->frames * (int64_t)spf);



    return 0;

}
