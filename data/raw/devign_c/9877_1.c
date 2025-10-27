static int decode_slice_luma(AVCodecContext *avctx, SliceContext *slice,

                             uint16_t *dst, int dst_stride,

                             const uint8_t *buf, unsigned buf_size,

                             const int16_t *qmat)

{

    ProresContext *ctx = avctx->priv_data;

    LOCAL_ALIGNED_16(int16_t, blocks, [8*4*64]);

    int16_t *block;

    GetBitContext gb;

    int i, blocks_per_slice = slice->mb_count<<2;

    int ret;



    for (i = 0; i < blocks_per_slice; i++)

        ctx->bdsp.clear_block(blocks+(i<<6));



    init_get_bits(&gb, buf, buf_size << 3);



    decode_dc_coeffs(&gb, blocks, blocks_per_slice);

    if ((ret = decode_ac_coeffs(avctx, &gb, blocks, blocks_per_slice)) < 0)

        return ret;



    block = blocks;

    for (i = 0; i < slice->mb_count; i++) {

        ctx->prodsp.idct_put(dst, dst_stride, block+(0<<6), qmat);

        ctx->prodsp.idct_put(dst             +8, dst_stride, block+(1<<6), qmat);

        ctx->prodsp.idct_put(dst+4*dst_stride  , dst_stride, block+(2<<6), qmat);

        ctx->prodsp.idct_put(dst+4*dst_stride+8, dst_stride, block+(3<<6), qmat);

        block += 4*64;

        dst += 16;

    }

    return 0;

}
