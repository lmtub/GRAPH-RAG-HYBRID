static av_cold int wmavoice_decode_init(AVCodecContext *ctx)

{

    int n, flags, pitch_range, lsp16_flag;

    WMAVoiceContext *s = ctx->priv_data;



    /**

     * Extradata layout:

     * - byte  0-18: WMAPro-in-WMAVoice extradata (see wmaprodec.c),

     * - byte 19-22: flags field (annoyingly in LE; see below for known

     *               values),

     * - byte 23-46: variable bitmode tree (really just 17 * 3 bits,

     *               rest is 0).

     */

    if (ctx->extradata_size != 46) {

        av_log(ctx, AV_LOG_ERROR,

               "Invalid extradata size %d (should be 46)\n",

               ctx->extradata_size);



    flags                = AV_RL32(ctx->extradata + 18);

    s->spillover_bitsize = 3 + av_ceil_log2(ctx->block_align);

    s->do_apf            =    flags & 0x1;

    if (s->do_apf) {

        ff_rdft_init(&s->rdft,  7, DFT_R2C);

        ff_rdft_init(&s->irdft, 7, IDFT_C2R);

        ff_dct_init(&s->dct,  6, DCT_I);

        ff_dct_init(&s->dst,  6, DST_I);



        ff_sine_window_init(s->cos, 256);

        memcpy(&s->sin[255], s->cos, 256 * sizeof(s->cos[0]));

        for (n = 0; n < 255; n++) {

            s->sin[n]       = -s->sin[510 - n];

            s->cos[510 - n] =  s->cos[n];



    s->denoise_strength  =   (flags >> 2) & 0xF;

    if (s->denoise_strength >= 12) {

        av_log(ctx, AV_LOG_ERROR,

               "Invalid denoise filter strength %d (max=11)\n",

               s->denoise_strength);



    s->denoise_tilt_corr = !!(flags & 0x40);

    s->dc_level          =   (flags >> 7) & 0xF;

    s->lsp_q_mode        = !!(flags & 0x2000);

    s->lsp_def_mode      = !!(flags & 0x4000);

    lsp16_flag           =    flags & 0x1000;

    if (lsp16_flag) {

        s->lsps               = 16;

        s->frame_lsp_bitsize  = 34;

        s->sframe_lsp_bitsize = 60;

    } else {

        s->lsps               = 10;

        s->frame_lsp_bitsize  = 24;

        s->sframe_lsp_bitsize = 48;


    for (n = 0; n < s->lsps; n++)

        s->prev_lsps[n] = M_PI * (n + 1.0) / (s->lsps + 1.0);



    init_get_bits(&s->gb, ctx->extradata + 22, (ctx->extradata_size - 22) << 3);

    if (decode_vbmtree(&s->gb, s->vbm_tree) < 0) {

        av_log(ctx, AV_LOG_ERROR, "Invalid VBM tree; broken extradata?\n");





    s->min_pitch_val    = ((ctx->sample_rate << 8)      /  400 + 50) >> 8;

    s->max_pitch_val    = ((ctx->sample_rate << 8) * 37 / 2000 + 50) >> 8;

    pitch_range         = s->max_pitch_val - s->min_pitch_val;

    if (pitch_range <= 0) {

        av_log(ctx, AV_LOG_ERROR, "Invalid pitch range; broken extradata?\n");



    s->pitch_nbits      = av_ceil_log2(pitch_range);

    s->last_pitch_val   = 40;

    s->last_acb_type    = ACB_TYPE_NONE;

    s->history_nsamples = s->max_pitch_val + 8;



    if (s->min_pitch_val < 1 || s->history_nsamples > MAX_SIGNAL_HISTORY) {

        int min_sr = ((((1 << 8) - 50) * 400) + 0xFF) >> 8,

            max_sr = ((((MAX_SIGNAL_HISTORY - 8) << 8) + 205) * 2000 / 37) >> 8;



        av_log(ctx, AV_LOG_ERROR,

               "Unsupported samplerate %d (min=%d, max=%d)\n",

               ctx->sample_rate, min_sr, max_sr); // 322-22097 Hz







    s->block_conv_table[0]      = s->min_pitch_val;

    s->block_conv_table[1]      = (pitch_range * 25) >> 6;

    s->block_conv_table[2]      = (pitch_range * 44) >> 6;

    s->block_conv_table[3]      = s->max_pitch_val - 1;

    s->block_delta_pitch_hrange = (pitch_range >> 3) & ~0xF;





    s->block_delta_pitch_nbits  = 1 + av_ceil_log2(s->block_delta_pitch_hrange);

    s->block_pitch_range        = s->block_conv_table[2] +

                                  s->block_conv_table[3] + 1 +

                                  2 * (s->block_conv_table[1] - 2 * s->min_pitch_val);

    s->block_pitch_nbits        = av_ceil_log2(s->block_pitch_range);



    ctx->sample_fmt             = AV_SAMPLE_FMT_FLT;



    return 0;
