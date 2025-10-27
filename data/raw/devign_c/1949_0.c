static void avc_luma_vt_8w_msa(const uint8_t *src, int32_t src_stride,

                               uint8_t *dst, int32_t dst_stride,

                               int32_t height)

{

    int32_t loop_cnt;

    int16_t filt_const0 = 0xfb01;

    int16_t filt_const1 = 0x1414;

    int16_t filt_const2 = 0x1fb;

    v16i8 src0, src1, src2, src3, src4, src7, src8, src9, src10;

    v16i8 src10_r, src32_r, src76_r, src98_r;

    v16i8 src21_r, src43_r, src87_r, src109_r;

    v8i16 out0_r, out1_r, out2_r, out3_r;

    v16i8 filt0, filt1, filt2;

    v16u8 out0, out1;



    filt0 = (v16i8) __msa_fill_h(filt_const0);

    filt1 = (v16i8) __msa_fill_h(filt_const1);

    filt2 = (v16i8) __msa_fill_h(filt_const2);



    LD_SB5(src, src_stride, src0, src1, src2, src3, src4);

    src += (5 * src_stride);



    XORI_B5_128_SB(src0, src1, src2, src3, src4);

    ILVR_B4_SB(src1, src0, src2, src1, src3, src2, src4, src3,

               src10_r, src21_r, src32_r, src43_r);



    for (loop_cnt = (height >> 2); loop_cnt--;) {

        LD_SB4(src, src_stride, src7, src8, src9, src10);

        src += (4 * src_stride);



        XORI_B4_128_SB(src7, src8, src9, src10);

        ILVR_B4_SB(src7, src4, src8, src7, src9, src8, src10, src9,

                   src76_r, src87_r, src98_r, src109_r);

        out0_r = DPADD_SH3_SH(src10_r, src32_r, src76_r, filt0, filt1, filt2);

        out1_r = DPADD_SH3_SH(src21_r, src43_r, src87_r, filt0, filt1, filt2);

        out2_r = DPADD_SH3_SH(src32_r, src76_r, src98_r, filt0, filt1, filt2);

        out3_r = DPADD_SH3_SH(src43_r, src87_r, src109_r, filt0, filt1, filt2);

        SRARI_H4_SH(out0_r, out1_r, out2_r, out3_r, 5);

        SAT_SH4_SH(out0_r, out1_r, out2_r, out3_r, 7);

        out0 = PCKEV_XORI128_UB(out0_r, out1_r);

        out1 = PCKEV_XORI128_UB(out2_r, out3_r);

        ST8x4_UB(out0, out1, dst, dst_stride);

        dst += (4 * dst_stride);



        src10_r = src76_r;

        src32_r = src98_r;

        src21_r = src87_r;

        src43_r = src109_r;

        src4 = src10;

    }

}
