static void avc_luma_vt_and_aver_dst_4x4_msa(const uint8_t *src,

                                             int32_t src_stride,

                                             uint8_t *dst, int32_t dst_stride)

{

    int16_t filt_const0 = 0xfb01;

    int16_t filt_const1 = 0x1414;

    int16_t filt_const2 = 0x1fb;

    v16u8 dst0, dst1, dst2, dst3;

    v16i8 src0, src1, src2, src3, src4, src5, src6, src7, src8;

    v16i8 src10_r, src32_r, src54_r, src76_r, src21_r, src43_r, src65_r;

    v16i8 src87_r, src2110, src4332, src6554, src8776;

    v8i16 out10, out32;

    v16i8 filt0, filt1, filt2;

    v16u8 res;



    filt0 = (v16i8) __msa_fill_h(filt_const0);

    filt1 = (v16i8) __msa_fill_h(filt_const1);

    filt2 = (v16i8) __msa_fill_h(filt_const2);



    LD_SB5(src, src_stride, src0, src1, src2, src3, src4);

    src += (5 * src_stride);



    ILVR_B4_SB(src1, src0, src2, src1, src3, src2, src4, src3,

               src10_r, src21_r, src32_r, src43_r);

    ILVR_D2_SB(src21_r, src10_r, src43_r, src32_r, src2110, src4332);

    XORI_B2_128_SB(src2110, src4332);

    LD_SB4(src, src_stride, src5, src6, src7, src8);

    ILVR_B4_SB(src5, src4, src6, src5, src7, src6, src8, src7,

               src54_r, src65_r, src76_r, src87_r);

    ILVR_D2_SB(src65_r, src54_r, src87_r, src76_r, src6554, src8776);

    XORI_B2_128_SB(src6554, src8776);

    out10 = DPADD_SH3_SH(src2110, src4332, src6554, filt0, filt1, filt2);

    out32 = DPADD_SH3_SH(src4332, src6554, src8776, filt0, filt1, filt2);

    SRARI_H2_SH(out10, out32, 5);

    SAT_SH2_SH(out10, out32, 7);

    LD_UB4(dst, dst_stride, dst0, dst1, dst2, dst3);

    res = PCKEV_XORI128_UB(out10, out32);



    ILVR_W2_UB(dst1, dst0, dst3, dst2, dst0, dst1);



    dst0 = (v16u8) __msa_pckev_d((v2i64) dst1, (v2i64) dst0);

    dst0 = __msa_aver_u_b(res, dst0);



    ST4x4_UB(dst0, dst0, 0, 1, 2, 3, dst, dst_stride);

}
