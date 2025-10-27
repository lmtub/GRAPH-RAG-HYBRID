static int decode_phys_chunk(AVCodecContext *avctx, PNGDecContext *s)

{

    if (s->state & PNG_IDAT) {

        av_log(avctx, AV_LOG_ERROR, "pHYs after IDAT\n");

        return AVERROR_INVALIDDATA;

    }

    avctx->sample_aspect_ratio.num = bytestream2_get_be32(&s->gb);

    avctx->sample_aspect_ratio.den = bytestream2_get_be32(&s->gb);

    if (avctx->sample_aspect_ratio.num < 0 || avctx->sample_aspect_ratio.den < 0)

        avctx->sample_aspect_ratio = (AVRational){ 0, 1 };

    bytestream2_skip(&s->gb, 1); /* unit specifier */

    bytestream2_skip(&s->gb, 4); /* crc */



    return 0;

}
