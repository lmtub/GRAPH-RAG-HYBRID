static void avc_luma_midv_qrt_4w_msa(const uint8_t *src, int32_t src_stride,

                                     uint8_t *dst, int32_t dst_stride,

                                     int32_t height, uint8_t ver_offset)

{

    uint32_t loop_cnt;

    v16i8 src0, src1, src2, src3, src4;

    v16i8 mask0, mask1, mask2;

    v8i16 hz_out0, hz_out1, hz_out2, hz_out3;

    v8i16 hz_out4, hz_out5, hz_out6, hz_out7, hz_out8;

    v8i16 dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;



    LD_SB3(&luma_mask_arr[48], 16, mask0, mask1, mask2);

    LD_SB5(src, src_stride, src0, src1, src2, src3, src4);

    src += (5 * src_stride);



    XORI_B5_128_SB(src0, src1, src2, src3, src4);



    hz_out0 = AVC_XOR_VSHF_B_AND_APPLY_6TAP_HORIZ_FILT_SH(src0, src1,

                                                          mask0, mask1, mask2);

    hz_out2 = AVC_XOR_VSHF_B_AND_APPLY_6TAP_HORIZ_FILT_SH(src2, src3,

                                                          mask0, mask1, mask2);



    PCKOD_D2_SH(hz_out0, hz_out0, hz_out2, hz_out2, hz_out1, hz_out3);



    hz_out4 = AVC_HORZ_FILTER_SH(src4, mask0, mask1, mask2);



    for (loop_cnt = (height >> 2); loop_cnt--;) {

        LD_SB4(src, src_stride, src0, src1, src2, src3);

        src += (4 * src_stride);

        XORI_B4_128_SB(src0, src1, src2, src3);



        hz_out5 = AVC_XOR_VSHF_B_AND_APPLY_6TAP_HORIZ_FILT_SH(src0, src1,

                                                              mask0, mask1,

                                                              mask2);

        hz_out7 = AVC_XOR_VSHF_B_AND_APPLY_6TAP_HORIZ_FILT_SH(src2, src3,

                                                              mask0, mask1,

                                                              mask2);



        PCKOD_D2_SH(hz_out5, hz_out5, hz_out7, hz_out7, hz_out6, hz_out8);



        dst0 = AVC_CALC_DPADD_H_6PIX_2COEFF_SH(hz_out0, hz_out1, hz_out2,

                                               hz_out3, hz_out4, hz_out5);

        dst2 = AVC_CALC_DPADD_H_6PIX_2COEFF_SH(hz_out1, hz_out2, hz_out3,

                                               hz_out4, hz_out5, hz_out6);

        dst4 = AVC_CALC_DPADD_H_6PIX_2COEFF_SH(hz_out2, hz_out3, hz_out4,

                                               hz_out5, hz_out6, hz_out7);

        dst6 = AVC_CALC_DPADD_H_6PIX_2COEFF_SH(hz_out3, hz_out4, hz_out5,

                                               hz_out6, hz_out7, hz_out8);



        if (ver_offset) {

            dst1 = __msa_srari_h(hz_out3, 5);

            dst3 = __msa_srari_h(hz_out4, 5);

            dst5 = __msa_srari_h(hz_out5, 5);

            dst7 = __msa_srari_h(hz_out6, 5);

        } else {

            dst1 = __msa_srari_h(hz_out2, 5);

            dst3 = __msa_srari_h(hz_out3, 5);

            dst5 = __msa_srari_h(hz_out4, 5);

            dst7 = __msa_srari_h(hz_out5, 5);

        }



        SAT_SH4_SH(dst1, dst3, dst5, dst7, 7);



        dst0 = __msa_aver_s_h(dst0, dst1);

        dst1 = __msa_aver_s_h(dst2, dst3);

        dst2 = __msa_aver_s_h(dst4, dst5);

        dst3 = __msa_aver_s_h(dst6, dst7);



        PCKEV_B2_SB(dst1, dst0, dst3, dst2, src0, src1);

        XORI_B2_128_SB(src0, src1);



        ST4x4_UB(src0, src1, 0, 2, 0, 2, dst, dst_stride);



        dst += (4 * dst_stride);

        hz_out0 = hz_out4;

        hz_out1 = hz_out5;

        hz_out2 = hz_out6;

        hz_out3 = hz_out7;

        hz_out4 = hz_out8;

    }

}
