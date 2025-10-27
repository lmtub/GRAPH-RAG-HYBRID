static int vpx_decode(AVCodecContext *avctx,
                      void *data, int *got_frame, AVPacket *avpkt)
{
    VPxContext *ctx = avctx->priv_data;
    AVFrame *picture = data;
    const void *iter = NULL;
    const void *iter_alpha = NULL;
    struct vpx_image *img, *img_alpha;
    int ret;
    uint8_t *side_data = NULL;
    int side_data_size = 0;
    ret = decode_frame(avctx, &ctx->decoder, avpkt->data, avpkt->size);
    if (ret)
        return ret;
    side_data = av_packet_get_side_data(avpkt,
                                        AV_PKT_DATA_MATROSKA_BLOCKADDITIONAL,
                                        &side_data_size);
    if (side_data_size > 1) {
        const uint64_t additional_id = AV_RB64(side_data);
        side_data += 8;
        side_data_size -= 8;
        if (additional_id == 1) {  // 1 stands for alpha channel data.
            if (!ctx->has_alpha_channel) {
                ctx->has_alpha_channel = 1;
                ret = vpx_init(avctx,
#if CONFIG_LIBVPX_VP8_DECODER && CONFIG_LIBVPX_VP9_DECODER
                               (avctx->codec_id == AV_CODEC_ID_VP8) ?
                               &vpx_codec_vp8_dx_algo : &vpx_codec_vp9_dx_algo,
#elif CONFIG_LIBVPX_VP8_DECODER
                               &vpx_codec_vp8_dx_algo,
#else
                               &vpx_codec_vp9_dx_algo,
#endif
                               1);
                if (ret)
                    return ret;
            ret = decode_frame(avctx, &ctx->decoder_alpha, side_data,
                               side_data_size);
            if (ret)
                return ret;
    if ((img = vpx_codec_get_frame(&ctx->decoder, &iter)) &&
        (!ctx->has_alpha_channel ||
         (img_alpha = vpx_codec_get_frame(&ctx->decoder_alpha, &iter_alpha)))) {
        uint8_t *planes[4];
        int linesizes[4];
        if ((ret = set_pix_fmt(avctx, img, ctx->has_alpha_channel)) < 0) {
#ifdef VPX_IMG_FMT_HIGHBITDEPTH
            av_log(avctx, AV_LOG_ERROR, "Unsupported output colorspace (%d) / bit_depth (%d)\n",
                   img->fmt, img->bit_depth);
#else
            av_log(avctx, AV_LOG_ERROR, "Unsupported output colorspace (%d) / bit_depth (%d)\n",
                   img->fmt, 8);
#endif
            return ret;
        if ((int) img->d_w != avctx->width || (int) img->d_h != avctx->height) {
            av_log(avctx, AV_LOG_INFO, "dimension change! %dx%d -> %dx%d\n",
                   avctx->width, avctx->height, img->d_w, img->d_h);
            ret = ff_set_dimensions(avctx, img->d_w, img->d_h);
            if (ret < 0)
                return ret;
        if ((ret = ff_get_buffer(avctx, picture, 0)) < 0)
            return ret;
        planes[0] = img->planes[VPX_PLANE_Y];
        planes[1] = img->planes[VPX_PLANE_U];
        planes[2] = img->planes[VPX_PLANE_V];
        planes[3] =
            ctx->has_alpha_channel ? img_alpha->planes[VPX_PLANE_Y] : NULL;
        linesizes[0] = img->stride[VPX_PLANE_Y];
        linesizes[1] = img->stride[VPX_PLANE_U];
        linesizes[2] = img->stride[VPX_PLANE_V];
        linesizes[3] =
            ctx->has_alpha_channel ? img_alpha->stride[VPX_PLANE_Y] : 0;
        av_image_copy(picture->data, picture->linesize, (const uint8_t**)planes,
                      linesizes, avctx->pix_fmt, img->d_w, img->d_h);
        *got_frame           = 1;
    return avpkt->size;