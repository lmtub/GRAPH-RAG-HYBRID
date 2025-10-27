static av_cold int vpx_init(AVCodecContext *avctx,
                            const struct vpx_codec_iface *iface)
{
    VP8Context *ctx = avctx->priv_data;
    struct vpx_codec_enc_cfg enccfg;
    int res;
    av_log(avctx, AV_LOG_INFO, "%s\n", vpx_codec_version_str());
    av_log(avctx, AV_LOG_VERBOSE, "%s\n", vpx_codec_build_config());
    if ((res = vpx_codec_enc_config_default(iface, &enccfg, 0)) != VPX_CODEC_OK) {
        av_log(avctx, AV_LOG_ERROR, "Failed to get config: %s\n",
               vpx_codec_err_to_string(res));
    if(!avctx->bit_rate)
        if(avctx->rc_max_rate || avctx->rc_buffer_size || avctx->rc_initial_buffer_occupancy) {
            av_log( avctx, AV_LOG_ERROR, "Rate control parameters set without a bitrate\n");
    dump_enc_cfg(avctx, &enccfg);
    enccfg.g_w            = avctx->width;
    enccfg.g_h            = avctx->height;
    enccfg.g_timebase.num = avctx->time_base.num;
    enccfg.g_timebase.den = avctx->time_base.den;
    enccfg.g_threads      = avctx->thread_count;
    enccfg.g_lag_in_frames= ctx->lag_in_frames;
    if (avctx->flags & CODEC_FLAG_PASS1)
        enccfg.g_pass = VPX_RC_FIRST_PASS;
    else if (avctx->flags & CODEC_FLAG_PASS2)
        enccfg.g_pass = VPX_RC_LAST_PASS;
    else
        enccfg.g_pass = VPX_RC_ONE_PASS;
    if (avctx->rc_min_rate == avctx->rc_max_rate &&
        avctx->rc_min_rate == avctx->bit_rate && avctx->bit_rate)
        enccfg.rc_end_usage = VPX_CBR;
    else if (ctx->crf)
        enccfg.rc_end_usage = VPX_CQ;
    if (avctx->bit_rate) {
        enccfg.rc_target_bitrate = av_rescale_rnd(avctx->bit_rate, 1, 1000,
                                                AV_ROUND_NEAR_INF);
    } else {
            enccfg.rc_target_bitrate = 1000000;
        } else {
            avctx->bit_rate = enccfg.rc_target_bitrate * 1000;
            av_log(avctx, AV_LOG_WARNING,
                   "Neither bitrate nor constrained quality specified, using default bitrate of %dkbit/sec\n",
                   enccfg.rc_target_bitrate);
    if (avctx->qmin >= 0)
        enccfg.rc_min_quantizer = avctx->qmin;
    if (avctx->qmax > 0)
        enccfg.rc_max_quantizer = avctx->qmax;
    enccfg.rc_dropframe_thresh = avctx->frame_skip_threshold;
    //0-100 (0 => CBR, 100 => VBR)
    enccfg.rc_2pass_vbr_bias_pct           = round(avctx->qcompress * 100);
    if (avctx->bit_rate)
        enccfg.rc_2pass_vbr_minsection_pct     =
            avctx->rc_min_rate * 100LL / avctx->bit_rate;
    if (avctx->rc_max_rate)
        enccfg.rc_2pass_vbr_maxsection_pct =
            avctx->rc_max_rate * 100LL / avctx->bit_rate;
    if (avctx->rc_buffer_size)
        enccfg.rc_buf_sz         =
            avctx->rc_buffer_size * 1000LL / avctx->bit_rate;
    if (avctx->rc_initial_buffer_occupancy)
        enccfg.rc_buf_initial_sz =
            avctx->rc_initial_buffer_occupancy * 1000LL / avctx->bit_rate;
    enccfg.rc_buf_optimal_sz     = enccfg.rc_buf_sz * 5 / 6;
    enccfg.rc_undershoot_pct     = round(avctx->rc_buffer_aggressivity * 100);
    //_enc_init() will balk if kf_min_dist differs from max w/VPX_KF_AUTO
    if (avctx->keyint_min >= 0 && avctx->keyint_min == avctx->gop_size)
        enccfg.kf_min_dist = avctx->keyint_min;
    if (avctx->gop_size >= 0)
        enccfg.kf_max_dist = avctx->gop_size;
    if (enccfg.g_pass == VPX_RC_FIRST_PASS)
        enccfg.g_lag_in_frames = 0;
    else if (enccfg.g_pass == VPX_RC_LAST_PASS) {
        int decode_size;
        if (!avctx->stats_in) {
            av_log(avctx, AV_LOG_ERROR, "No stats file for second pass\n");
            return AVERROR_INVALIDDATA;
        ctx->twopass_stats.sz  = strlen(avctx->stats_in) * 3 / 4;
        ctx->twopass_stats.buf = av_malloc(ctx->twopass_stats.sz);
        if (!ctx->twopass_stats.buf) {
                   "Stat buffer alloc (%zu bytes) failed\n",
                   ctx->twopass_stats.sz);
            return AVERROR(ENOMEM);
        decode_size = av_base64_decode(ctx->twopass_stats.buf, avctx->stats_in,
                                       ctx->twopass_stats.sz);
        if (decode_size < 0) {
            av_log(avctx, AV_LOG_ERROR, "Stat buffer decode failed\n");
            return AVERROR_INVALIDDATA;
        ctx->twopass_stats.sz      = decode_size;
        enccfg.rc_twopass_stats_in = ctx->twopass_stats;
    /* 0-3: For non-zero values the encoder increasingly optimizes for reduced
       complexity playback on low powered devices at the expense of encode
       quality. */
   if (avctx->profile != FF_PROFILE_UNKNOWN)
       enccfg.g_profile = avctx->profile;
    enccfg.g_error_resilient = ctx->error_resilient || ctx->flags & VP8F_ERROR_RESILIENT;
    dump_enc_cfg(avctx, &enccfg);
    /* Construct Encoder Context */
    res = vpx_codec_enc_init(&ctx->encoder, iface, &enccfg, 0);
    if (res != VPX_CODEC_OK) {
        log_encoder_error(avctx, "Failed to initialize encoder");
    //codec control failures are currently treated only as warnings
    av_log(avctx, AV_LOG_DEBUG, "vpx_codec_control\n");
    if (ctx->cpu_used != INT_MIN)
        codecctl_int(avctx, VP8E_SET_CPUUSED,          ctx->cpu_used);
    if (ctx->flags & VP8F_AUTO_ALT_REF)
        ctx->auto_alt_ref = 1;
    if (ctx->auto_alt_ref >= 0)
        codecctl_int(avctx, VP8E_SET_ENABLEAUTOALTREF, ctx->auto_alt_ref);
    if (ctx->arnr_max_frames >= 0)
        codecctl_int(avctx, VP8E_SET_ARNR_MAXFRAMES,   ctx->arnr_max_frames);
    if (ctx->arnr_strength >= 0)
        codecctl_int(avctx, VP8E_SET_ARNR_STRENGTH,    ctx->arnr_strength);
    if (ctx->arnr_type >= 0)
        codecctl_int(avctx, VP8E_SET_ARNR_TYPE,        ctx->arnr_type);
    codecctl_int(avctx, VP8E_SET_NOISE_SENSITIVITY, avctx->noise_reduction);
    codecctl_int(avctx, VP8E_SET_TOKEN_PARTITIONS,  av_log2(avctx->slices));
    codecctl_int(avctx, VP8E_SET_STATIC_THRESHOLD,  avctx->mb_threshold);
    codecctl_int(avctx, VP8E_SET_CQ_LEVEL,          ctx->crf);
    if (ctx->max_intra_rate >= 0)
        codecctl_int(avctx, VP8E_SET_MAX_INTRA_BITRATE_PCT, ctx->max_intra_rate);
    av_log(avctx, AV_LOG_DEBUG, "Using deadline: %d\n", ctx->deadline);
    //provide dummy value to initialize wrapper, values will be updated each _encode()
    vpx_img_wrap(&ctx->rawimg, VPX_IMG_FMT_I420, avctx->width, avctx->height, 1,
                 (unsigned char*)1);
    avctx->coded_frame = avcodec_alloc_frame();
    if (!avctx->coded_frame) {
        av_log(avctx, AV_LOG_ERROR, "Error allocating coded frame\n");
        vp8_free(avctx);
        return AVERROR(ENOMEM);
    return 0;