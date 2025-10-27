static int r3d_read_red1(AVFormatContext *s)

{

    AVStream *st = avformat_new_stream(s, NULL);

    char filename[258];

    int tmp;

    int av_unused tmp2;

    AVRational framerate;



    if (!st)

        return AVERROR(ENOMEM);

    st->codec->codec_type = AVMEDIA_TYPE_VIDEO;

    st->codec->codec_id = CODEC_ID_JPEG2000;



    tmp  = avio_r8(s->pb); // major version

    tmp2 = avio_r8(s->pb); // minor version

    av_dlog(s, "version %d.%d\n", tmp, tmp2);



    tmp = avio_rb16(s->pb); // unknown

    av_dlog(s, "unknown1 %d\n", tmp);



    tmp = avio_rb32(s->pb);

    avpriv_set_pts_info(st, 32, 1, tmp);



    tmp = avio_rb32(s->pb); // filenum

    av_dlog(s, "filenum %d\n", tmp);



    avio_skip(s->pb, 32); // unknown



    st->codec->width  = avio_rb32(s->pb);

    st->codec->height = avio_rb32(s->pb);



    tmp = avio_rb16(s->pb); // unknown

    av_dlog(s, "unknown2 %d\n", tmp);



    framerate.num = avio_rb16(s->pb);

    framerate.den = avio_rb16(s->pb);

    if (framerate.num && framerate.den)

        st->r_frame_rate = st->avg_frame_rate = framerate;



    tmp = avio_r8(s->pb); // audio channels

    av_dlog(s, "audio channels %d\n", tmp);

    if (tmp > 0) {

        AVStream *ast = avformat_new_stream(s, NULL);

        if (!ast)

            return AVERROR(ENOMEM);

        ast->codec->codec_type = AVMEDIA_TYPE_AUDIO;

        ast->codec->codec_id = CODEC_ID_PCM_S32BE;

        ast->codec->channels = tmp;

        avpriv_set_pts_info(ast, 32, 1, st->time_base.den);

    }



    avio_read(s->pb, filename, 257);

    filename[sizeof(filename)-1] = 0;

    av_dict_set(&st->metadata, "filename", filename, 0);



    av_dlog(s, "filename %s\n", filename);

    av_dlog(s, "resolution %dx%d\n", st->codec->width, st->codec->height);

    av_dlog(s, "timescale %d\n", st->time_base.den);

    av_dlog(s, "frame rate %d/%d\n",

            framerate.num, framerate.den);



    return 0;

}
