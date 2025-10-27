static int nuv_header(AVFormatContext *s)

{

    NUVContext *ctx = s->priv_data;

    AVIOContext *pb = s->pb;

    char id_string[12];

    double aspect, fps;

    int is_mythtv, width, height, v_packs, a_packs;

    AVStream *vst = NULL, *ast = NULL;



    avio_read(pb, id_string, 12);

    is_mythtv = !memcmp(id_string, "MythTVVideo", 12);

    avio_skip(pb, 5);       // version string

    avio_skip(pb, 3);       // padding

    width  = avio_rl32(pb);

    height = avio_rl32(pb);

    avio_rl32(pb);          // unused, "desiredwidth"

    avio_rl32(pb);          // unused, "desiredheight"

    avio_r8(pb);            // 'P' == progressive, 'I' == interlaced

    avio_skip(pb, 3);       // padding

    aspect = av_int2double(avio_rl64(pb));

    if (aspect > 0.9999 && aspect < 1.0001)

        aspect = 4.0 / 3.0;

    fps = av_int2double(avio_rl64(pb));



    // number of packets per stream type, -1 means unknown, e.g. streaming

    v_packs = avio_rl32(pb);

    a_packs = avio_rl32(pb);

    avio_rl32(pb); // text



    avio_rl32(pb); // keyframe distance (?)



    if (v_packs) {

        vst = avformat_new_stream(s, NULL);

        if (!vst)

            return AVERROR(ENOMEM);

        ctx->v_id = vst->index;



        vst->codec->codec_type            = AVMEDIA_TYPE_VIDEO;

        vst->codec->codec_id              = AV_CODEC_ID_NUV;

        vst->codec->width                 = width;

        vst->codec->height                = height;

        vst->codec->bits_per_coded_sample = 10;

        vst->sample_aspect_ratio          = av_d2q(aspect * height / width,

                                                   10000);

#if FF_API_R_FRAME_RATE

        vst->r_frame_rate =

#endif

        vst->avg_frame_rate = av_d2q(fps, 60000);

        avpriv_set_pts_info(vst, 32, 1, 1000);

    } else

        ctx->v_id = -1;



    if (a_packs) {

        ast = avformat_new_stream(s, NULL);

        if (!ast)

            return AVERROR(ENOMEM);

        ctx->a_id = ast->index;



        ast->codec->codec_type            = AVMEDIA_TYPE_AUDIO;

        ast->codec->codec_id              = AV_CODEC_ID_PCM_S16LE;

        ast->codec->channels              = 2;

        ast->codec->channel_layout        = AV_CH_LAYOUT_STEREO;

        ast->codec->sample_rate           = 44100;

        ast->codec->bit_rate              = 2 * 2 * 44100 * 8;

        ast->codec->block_align           = 2 * 2;

        ast->codec->bits_per_coded_sample = 16;

        avpriv_set_pts_info(ast, 32, 1, 1000);

    } else

        ctx->a_id = -1;



    get_codec_data(pb, vst, ast, is_mythtv);

    ctx->rtjpg_video = vst && vst->codec->codec_id == AV_CODEC_ID_NUV;



    return 0;

}
