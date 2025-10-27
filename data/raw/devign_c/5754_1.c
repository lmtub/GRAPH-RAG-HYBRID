static av_cold int roq_encode_init(AVCodecContext *avctx)
{
    RoqContext *enc = avctx->priv_data;
    av_lfg_init(&enc->randctx, 1);
    enc->framesSinceKeyframe = 0;
    if ((avctx->width & 0xf) || (avctx->height & 0xf)) {
        av_log(avctx, AV_LOG_ERROR, "Dimensions must be divisible by 16\n");
        return AVERROR(EINVAL);
    if (avctx->width > 65535 || avctx->height > 65535) {
        av_log(avctx, AV_LOG_ERROR, "Dimensions are max %d\n", enc->quake3_compat ? 32768 : 65535);
        return AVERROR(EINVAL);
    if (((avctx->width)&(avctx->width-1))||((avctx->height)&(avctx->height-1)))
        av_log(avctx, AV_LOG_ERROR, "Warning: dimensions not power of two, this is not supported by quake\n");
    enc->width = avctx->width;
    enc->height = avctx->height;
    enc->framesSinceKeyframe = 0;
    enc->first_frame = 1;
    enc->last_frame    = av_frame_alloc();
    enc->current_frame = av_frame_alloc();
    if (!enc->last_frame || !enc->current_frame) {
    enc->tmpData      = av_malloc(sizeof(RoqTempdata));
    enc->this_motion4 =
        av_mallocz_array((enc->width*enc->height/16), sizeof(motion_vect));
    enc->last_motion4 =
        av_malloc_array ((enc->width*enc->height/16), sizeof(motion_vect));
    enc->this_motion8 =
        av_mallocz_array((enc->width*enc->height/64), sizeof(motion_vect));
    enc->last_motion8 =
        av_malloc_array ((enc->width*enc->height/64), sizeof(motion_vect));
    return 0;