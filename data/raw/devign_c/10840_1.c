static int read_highpass(AVCodecContext *avctx, uint8_t *ptr, int plane, AVFrame *frame)
{
    PixletContext *ctx = avctx->priv_data;
    ptrdiff_t stride = frame->linesize[plane] / 2;
    int i, ret;
    for (i = 0; i < ctx->levels * 3; i++) {
        int32_t a = bytestream2_get_be32(&ctx->gb);
        int32_t b = bytestream2_get_be32(&ctx->gb);
        int32_t c = bytestream2_get_be32(&ctx->gb);
        int32_t d = bytestream2_get_be32(&ctx->gb);
        int16_t *dest = (int16_t *)frame->data[plane] + ctx->band[plane][i + 1].x +
                                               stride * ctx->band[plane][i + 1].y;
        unsigned size = ctx->band[plane][i + 1].size;
        uint32_t magic;
        magic = bytestream2_get_be32(&ctx->gb);
        if (magic != 0xDEADBEEF) {
            av_log(avctx, AV_LOG_ERROR, "wrong magic number: 0x%08"PRIX32
                   " for plane %d, band %d\n", magic, plane, i);
        }
        ret = read_high_coeffs(avctx, ptr + bytestream2_tell(&ctx->gb), dest, size,
                               c, (b >= FFABS(a)) ? b : a, d,
                               ctx->band[plane][i + 1].width, stride);
        if (ret < 0) {
            av_log(avctx, AV_LOG_ERROR, "error in highpass coefficients for plane %d, band %d\n", plane, i);
            return ret;
        }
        bytestream2_skip(&ctx->gb, ret);
    }
    return 0;
}