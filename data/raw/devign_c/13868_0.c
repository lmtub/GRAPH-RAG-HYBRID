static void intra_predict_hor_dc_8x8_msa(uint8_t *src, int32_t stride)

{

    uint8_t lp_cnt;

    uint32_t src0 = 0, src1 = 0;

    uint64_t out0, out1;



    for (lp_cnt = 0; lp_cnt < 4; lp_cnt++) {

        src0 += src[lp_cnt * stride - 1];

        src1 += src[(4 + lp_cnt) * stride - 1];

    }



    src0 = (src0 + 2) >> 2;

    src1 = (src1 + 2) >> 2;

    out0 = src0 * 0x0101010101010101;

    out1 = src1 * 0x0101010101010101;



    for (lp_cnt = 4; lp_cnt--;) {

        SD(out0, src);

        SD(out1, (src + 4 * stride));

        src += stride;

    }

}
