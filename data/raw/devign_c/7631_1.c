static int process_video_header_vp6(AVFormatContext *s)

{

    EaDemuxContext *ea = s->priv_data;

    AVIOContext *pb = s->pb;



    avio_skip(pb, 8);

    ea->nb_frames = avio_rl32(pb);

    avio_skip(pb, 4);

    ea->time_base.den = avio_rl32(pb);

    ea->time_base.num = avio_rl32(pb);





    ea->video_codec = AV_CODEC_ID_VP6;



    return 1;
