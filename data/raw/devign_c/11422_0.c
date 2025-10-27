static av_cold int hap_init(AVCodecContext *avctx)

{

    HapContext *ctx = avctx->priv_data;

    int ratio;

    int corrected_chunk_count;

    int ret = av_image_check_size(avctx->width, avctx->height, 0, avctx);



    if (ret < 0) {

        av_log(avctx, AV_LOG_ERROR, "Invalid video size %dx%d.\n",

               avctx->width, avctx->height);

        return ret;

    }



    if (avctx->width % 4 || avctx->height % 4) {

        av_log(avctx, AV_LOG_ERROR, "Video size %dx%d is not multiple of 4.\n",

               avctx->width, avctx->height);

        return AVERROR_INVALIDDATA;

    }



    ff_texturedspenc_init(&ctx->dxtc);



    switch (ctx->opt_tex_fmt) {

    case HAP_FMT_RGBDXT1:

        ratio = 8;

        avctx->codec_tag = MKTAG('H', 'a', 'p', '1');

        avctx->bits_per_coded_sample = 24;

        ctx->tex_fun = ctx->dxtc.dxt1_block;

        break;

    case HAP_FMT_RGBADXT5:

        ratio = 4;

        avctx->codec_tag = MKTAG('H', 'a', 'p', '5');

        avctx->bits_per_coded_sample = 32;

        ctx->tex_fun = ctx->dxtc.dxt5_block;

        break;

    case HAP_FMT_YCOCGDXT5:

        ratio = 4;

        avctx->codec_tag = MKTAG('H', 'a', 'p', 'Y');

        avctx->bits_per_coded_sample = 24;

        ctx->tex_fun = ctx->dxtc.dxt5ys_block;

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "Invalid format %02X\n", ctx->opt_tex_fmt);

        return AVERROR_INVALIDDATA;

    }



    /* Texture compression ratio is constant, so can we computer

     * beforehand the final size of the uncompressed buffer. */

    ctx->tex_size   = FFALIGN(avctx->width,  TEXTURE_BLOCK_W) *

                      FFALIGN(avctx->height, TEXTURE_BLOCK_H) * 4 / ratio;



    /* Round the chunk count to divide evenly on DXT block edges */

    corrected_chunk_count = av_clip(ctx->opt_chunk_count, 1, HAP_MAX_CHUNKS);

    while ((ctx->tex_size / (64 / ratio)) % corrected_chunk_count != 0) {

        corrected_chunk_count--;

    }

    if (corrected_chunk_count != ctx->opt_chunk_count) {

        av_log(avctx, AV_LOG_INFO, "%d chunks requested but %d used.\n",

                                    ctx->opt_chunk_count, corrected_chunk_count);

    }

    ret = ff_hap_set_chunk_count(ctx, corrected_chunk_count, 1);

    if (ret != 0)

        return ret;



    ctx->max_snappy = snappy_max_compressed_length(ctx->tex_size / corrected_chunk_count);



    ctx->tex_buf  = av_malloc(ctx->tex_size);

    if (!ctx->tex_buf)

        return AVERROR(ENOMEM);



    return 0;

}
