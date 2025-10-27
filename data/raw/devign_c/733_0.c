static void intra_predict_mad_cow_dc_l0t_8x8_msa(uint8_t *src, int32_t stride)

{

    uint8_t lp_cnt;

    uint32_t src0, src1, src2 = 0;

    uint32_t out0, out1, out2;

    v16u8 src_top;

    v8u16 add;

    v4u32 sum;



    src_top = LD_UB(src - stride);

    add = __msa_hadd_u_h(src_top, src_top);

    sum = __msa_hadd_u_w(add, add);

    src0 = __msa_copy_u_w((v4i32) sum, 0);

    src1 = __msa_copy_u_w((v4i32) sum, 1);



    for (lp_cnt = 0; lp_cnt < 4; lp_cnt++) {

        src2 += src[lp_cnt * stride - 1];

    }

    src2 = (src0 + src2 + 4) >> 3;

    src0 = (src0 + 2) >> 2;

    src1 = (src1 + 2) >> 2;

    out0 = src0 * 0x01010101;

    out1 = src1 * 0x01010101;

    out2 = src2 * 0x01010101;



    for (lp_cnt = 4; lp_cnt--;) {

        SW(out2, src);

        SW(out1, src + 4);

        SW(out0, src + stride * 4);

        SW(out1, src + stride * 4 + 4);

        src += stride;

    }

}
