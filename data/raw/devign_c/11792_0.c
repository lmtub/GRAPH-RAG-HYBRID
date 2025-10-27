static int vc1t_read_header(AVFormatContext *s,

                           AVFormatParameters *ap)

{

    ByteIOContext *pb = s->pb;

    AVStream *st;

    int fps, frames;



    frames = get_le24(pb);

    if(get_byte(pb) != 0xC5 || get_le32(pb) != 4)

        return -1;



    /* init video codec */

    st = av_new_stream(s, 0);

    if (!st)

        return -1;



    st->codec->codec_type = CODEC_TYPE_VIDEO;

    st->codec->codec_id = CODEC_ID_WMV3;



    st->codec->extradata = av_malloc(VC1_EXTRADATA_SIZE);

    st->codec->extradata_size = VC1_EXTRADATA_SIZE;

    get_buffer(pb, st->codec->extradata, VC1_EXTRADATA_SIZE);

    st->codec->height = get_le32(pb);

    st->codec->width = get_le32(pb);

    if(get_le32(pb) != 0xC)

        return -1;

    url_fskip(pb, 8);

    fps = get_le32(pb);

    if(fps == -1)

        av_set_pts_info(st, 32, 1, 1000);

    else{

        av_set_pts_info(st, 24, 1, fps);

        st->duration = frames;

    }



    return 0;

}
