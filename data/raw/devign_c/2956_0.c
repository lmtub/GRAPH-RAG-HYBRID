static int nuv_header(AVFormatContext *s)

{

    NUVContext *ctx = s->priv_data;

    AVIOContext *pb = s->pb;

    char id_string[12];

    double aspect, fps;

    int is_mythtv, width, height, v_packs, a_packs, ret;

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

    if (fps < 0.0f) {

        if (s->error_recognition & AV_EF_EXPLODE) {

            av_log(s, AV_LOG_ERROR, "Invalid frame rate %f\n", fps);

            return AVERROR_INVALIDDATA;

        } else {

            av_log(s, AV_LOG_WARNING, "Invalid frame rate %f, setting to 0.\n", fps);

            fps = 0.0f;

        }

    }



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



        ret = av_image_check_size(width, height, 0, s);

        if (ret < 0)

            return ret;



        vst->codecpar->codec_type            = AVMEDIA_TYPE_VIDEO;

        vst->codecpar->codec_id              = AV_CODEC_ID_NUV;

        vst->codecpar->width                 = width;

        vst->codecpar->height                = height;

        vst->codecpar->bits_per_coded_sample = 10;

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



        ast->codecpar->codec_type            = AVMEDIA_TYPE_AUDIO;

        ast->codecpar->codec_id              = AV_CODEC_ID_PCM_S16LE;

        ast->codecpar->channels              = 2;

        ast->codecpar->channel_layout        = AV_CH_LAYOUT_STEREO;

        ast->codecpar->sample_rate           = 44100;

        ast->codecpar->bit_rate              = 2 * 2 * 44100 * 8;

        ast->codecpar->block_align           = 2 * 2;

        ast->codecpar->bits_per_coded_sample = 16;

        avpriv_set_pts_info(ast, 32, 1, 1000);

    } else

        ctx->a_id = -1;



    if ((ret = get_codec_data(pb, vst, ast, is_mythtv)) < 0)

        return ret;



    ctx->rtjpg_video = vst && vst->codecpar->codec_id == AV_CODEC_ID_NUV;



    return 0;

}
