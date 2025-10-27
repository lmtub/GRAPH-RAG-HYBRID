int dxva2_init(AVCodecContext *s)

{

    InputStream *ist = s->opaque;

    int loglevel = (ist->hwaccel_id == HWACCEL_AUTO) ? AV_LOG_VERBOSE : AV_LOG_ERROR;

    DXVA2Context *ctx;

    int ret;



    if (!ist->hwaccel_ctx) {

        ret = dxva2_alloc(s);

        if (ret < 0)

            return ret;

    }

    ctx = ist->hwaccel_ctx;



    if (s->codec_id == AV_CODEC_ID_H264 &&

        (s->profile & ~FF_PROFILE_H264_CONSTRAINED) > FF_PROFILE_H264_HIGH) {

        av_log(NULL, loglevel, "Unsupported H.264 profile for DXVA2 HWAccel: %d\n", s->profile);

        return AVERROR(EINVAL);

    }



    if (s->codec_id == AV_CODEC_ID_HEVC &&

        s->profile != FF_PROFILE_HEVC_MAIN && s->profile != FF_PROFILE_HEVC_MAIN_10) {

        av_log(NULL, loglevel, "Unsupported HEVC profile for DXVA2 HWAccel: %d\n", s->profile);

        return AVERROR(EINVAL);

    }



    av_buffer_unref(&ctx->hw_frames_ctx);



    ret = dxva2_create_decoder(s);

    if (ret < 0) {

        av_log(NULL, loglevel, "Error creating the DXVA2 decoder\n");

        return ret;

    }



    return 0;

}
