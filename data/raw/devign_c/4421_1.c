static int encode_slice_plane(AVCodecContext *avctx, int mb_count,

        uint8_t *src, int src_stride, uint8_t *buf, unsigned buf_size,

        int *qmat, int chroma)

{

    ProresContext* ctx = avctx->priv_data;

    FDCTDSPContext *fdsp = &ctx->fdsp;

    DECLARE_ALIGNED(16, int16_t, blocks)[DEFAULT_SLICE_MB_WIDTH << 8], *block;

    int i, blocks_per_slice;

    PutBitContext pb;



    block = blocks;

    for (i = 0; i < mb_count; i++) {

        fdct_get(fdsp, src,                  src_stride, block + (0 << 6));

        fdct_get(fdsp, src + 8 * src_stride, src_stride, block + ((2 - chroma) << 6));

        if (!chroma) {

            fdct_get(fdsp, src + 16,                  src_stride, block + (1 << 6));

            fdct_get(fdsp, src + 16 + 8 * src_stride, src_stride, block + (3 << 6));

        }



        block += (256 >> chroma);

        src   += (32  >> chroma);

    }



    blocks_per_slice = mb_count << (2 - chroma);

    init_put_bits(&pb, buf, buf_size << 3);



    encode_dc_coeffs(&pb, blocks, blocks_per_slice, qmat);

    encode_ac_coeffs(avctx, &pb, blocks, blocks_per_slice, qmat);



    flush_put_bits(&pb);

    return put_bits_ptr(&pb) - pb.buf;

}
