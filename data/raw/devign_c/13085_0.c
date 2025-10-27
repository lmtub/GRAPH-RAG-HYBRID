static int mov_write_ftyp_tag(AVIOContext *pb, AVFormatContext *s)

{

    MOVMuxContext *mov = s->priv_data;

    int64_t pos = avio_tell(pb);

    int has_h264 = 0, has_video = 0;

    int minor = 0x200;

    int i;



    for (i = 0; i < s->nb_streams; i++) {

        AVStream *st = s->streams[i];

        if (st->codec->codec_type == AVMEDIA_TYPE_VIDEO)

            has_video = 1;

        if (st->codec->codec_id == AV_CODEC_ID_H264)

            has_h264 = 1;

    }



    avio_wb32(pb, 0); /* size */

    ffio_wfourcc(pb, "ftyp");



    if (mov->mode == MODE_3GP) {

        ffio_wfourcc(pb, has_h264 ? "3gp6"  : "3gp4");

        minor =     has_h264 ?   0x100 :   0x200;

    } else if (mov->mode & MODE_3G2) {

        ffio_wfourcc(pb, has_h264 ? "3g2b"  : "3g2a");

        minor =     has_h264 ? 0x20000 : 0x10000;

    } else if (mov->mode == MODE_PSP)

        ffio_wfourcc(pb, "MSNV");

    else if (mov->mode == MODE_MP4)

        ffio_wfourcc(pb, "isom");

    else if (mov->mode == MODE_IPOD)

        ffio_wfourcc(pb, has_video ? "M4V ":"M4A ");

    else if (mov->mode == MODE_ISM)

        ffio_wfourcc(pb, "isml");

    else if (mov->mode == MODE_F4V)

        ffio_wfourcc(pb, "f4v ");

    else

        ffio_wfourcc(pb, "qt  ");



    avio_wb32(pb, minor);



    if (mov->mode == MODE_MOV)

        ffio_wfourcc(pb, "qt  ");

    else if (mov->mode == MODE_ISM) {

        ffio_wfourcc(pb, "piff");

        ffio_wfourcc(pb, "iso2");

    } else {

        ffio_wfourcc(pb, "isom");

        ffio_wfourcc(pb, "iso2");

        if (has_h264)

            ffio_wfourcc(pb, "avc1");

    }



    if (mov->mode == MODE_3GP)

        ffio_wfourcc(pb, has_h264 ? "3gp6":"3gp4");

    else if (mov->mode & MODE_3G2)

        ffio_wfourcc(pb, has_h264 ? "3g2b":"3g2a");

    else if (mov->mode == MODE_PSP)

        ffio_wfourcc(pb, "MSNV");

    else if (mov->mode == MODE_MP4)

        ffio_wfourcc(pb, "mp41");

    return update_size(pb, pos);

}
