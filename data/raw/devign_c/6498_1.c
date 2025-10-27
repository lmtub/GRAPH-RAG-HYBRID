static int mtv_read_header(AVFormatContext *s)

{

    MTVDemuxContext *mtv = s->priv_data;

    AVIOContext   *pb  = s->pb;

    AVStream        *st;

    unsigned int    audio_subsegments;



    avio_skip(pb, 3);

    mtv->file_size         = avio_rl32(pb);

    mtv->segments          = avio_rl32(pb);

    avio_skip(pb, 32);

    mtv->audio_identifier  = avio_rl24(pb);

    mtv->audio_br          = avio_rl16(pb);

    mtv->img_colorfmt      = avio_rl24(pb);

    mtv->img_bpp           = avio_r8(pb);

    mtv->img_width         = avio_rl16(pb);

    mtv->img_height        = avio_rl16(pb);

    mtv->img_segment_size  = avio_rl16(pb);



    /* Calculate width and height if missing from header */



    if(!mtv->img_width)

        mtv->img_width=mtv->img_segment_size / (mtv->img_bpp>>3)

                        / mtv->img_height;



    if(!mtv->img_height)

        mtv->img_height=mtv->img_segment_size / (mtv->img_bpp>>3)

                        / mtv->img_width;



    avio_skip(pb, 4);

    audio_subsegments = avio_rl16(pb);



    if (audio_subsegments == 0) {

        avpriv_request_sample(s, "MTV files without audio");

        return AVERROR_PATCHWELCOME;

    }



    mtv->full_segment_size =

        audio_subsegments * (MTV_AUDIO_PADDING_SIZE + MTV_ASUBCHUNK_DATA_SIZE) +

        mtv->img_segment_size;

    mtv->video_fps         = (mtv->audio_br / 4) / audio_subsegments;



    // FIXME Add sanity check here



    // all systems go! init decoders



    // video - raw rgb565



    st = avformat_new_stream(s, NULL);

    if(!st)

        return AVERROR(ENOMEM);



    avpriv_set_pts_info(st, 64, 1, mtv->video_fps);

    st->codec->codec_type      = AVMEDIA_TYPE_VIDEO;

    st->codec->codec_id        = AV_CODEC_ID_RAWVIDEO;

    st->codec->pix_fmt         = AV_PIX_FMT_RGB565BE;

    st->codec->width           = mtv->img_width;

    st->codec->height          = mtv->img_height;

    st->codec->extradata       = av_strdup("BottomUp");

    st->codec->extradata_size  = 9;



    // audio - mp3



    st = avformat_new_stream(s, NULL);

    if(!st)

        return AVERROR(ENOMEM);



    avpriv_set_pts_info(st, 64, 1, AUDIO_SAMPLING_RATE);

    st->codec->codec_type      = AVMEDIA_TYPE_AUDIO;

    st->codec->codec_id        = AV_CODEC_ID_MP3;

    st->codec->bit_rate        = mtv->audio_br;

    st->need_parsing           = AVSTREAM_PARSE_FULL;



    // Jump over header



    if(avio_seek(pb, MTV_HEADER_SIZE, SEEK_SET) != MTV_HEADER_SIZE)

        return AVERROR(EIO);



    return 0;



}
