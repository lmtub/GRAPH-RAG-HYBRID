static void avc_luma_midv_qrt_and_aver_dst_8w_msa(const uint8_t *src,

                                                  int32_t src_stride,

                                                  uint8_t *dst,

                                                  int32_t dst_stride,

                                                  int32_t height,

                                                  uint8_t vert_offset)

{

    int32_t loop_cnt;

    v16i8 src0, src1, src2, src3, src4;

    v16u8 dst0, dst1, dst2, dst3;

    v16i8 mask0, mask1, mask2;

    v8i16 hz_out0, hz_out1, hz_out2, hz_out3;

    v8i16 hz_out4, hz_out5, hz_out6, hz_out7, hz_out8;

    v8i16 res0, res1, res2, res3;

    v8i16 res4, res5, res6, res7;



    LD_SB3(&luma_mask_arr[0], 16, mask0, mask1, mask2);



    LD_SB5(src, src_stride, src0, src1, src2, src3, src4);

    XORI_B5_128_SB(src0, src1, src2, src3, src4);

    src += (5 * src_stride);



    hz_out0 = AVC_HORZ_FILTER_SH(src0, mask0, mask1, mask2);

    hz_out1 = AVC_HORZ_FILTER_SH(src1, mask0, mask1, mask2);

    hz_out2 = AVC_HORZ_FILTER_SH(src2, mask0, mask1, mask2);

    hz_out3 = AVC_HORZ_FILTER_SH(src3, mask0, mask1, mask2);

    hz_out4 = AVC_HORZ_FILTER_SH(src4, mask0, mask1, mask2);



    for (loop_cnt = (height >> 2); loop_cnt--;) {

        LD_SB4(src, src_stride, src0, src1, src2, src3);

        XORI_B4_128_SB(src0, src1, src2, src3);

        src += (4 * src_stride);



        LD_UB4(dst, dst_stride, dst0, dst1, dst2, dst3);



        hz_out5 = AVC_HORZ_FILTER_SH(src0, mask0, mask1, mask2);

        hz_out6 = AVC_HORZ_FILTER_SH(src1, mask0, mask1, mask2);

        hz_out7 = AVC_HORZ_FILTER_SH(src2, mask0, mask1, mask2);

        hz_out8 = AVC_HORZ_FILTER_SH(src3, mask0, mask1, mask2);



        res0 = AVC_CALC_DPADD_H_6PIX_2COEFF_SH(hz_out0, hz_out1, hz_out2,

                                               hz_out3, hz_out4, hz_out5);

        res2 = AVC_CALC_DPADD_H_6PIX_2COEFF_SH(hz_out1, hz_out2, hz_out3,

                                               hz_out4, hz_out5, hz_out6);

        res4 = AVC_CALC_DPADD_H_6PIX_2COEFF_SH(hz_out2, hz_out3, hz_out4,

                                               hz_out5, hz_out6, hz_out7);

        res6 = AVC_CALC_DPADD_H_6PIX_2COEFF_SH(hz_out3, hz_out4, hz_out5,

                                               hz_out6, hz_out7, hz_out8);



        if (vert_offset) {

            res1 = __msa_srari_h(hz_out3, 5);

            res3 = __msa_srari_h(hz_out4, 5);

            res5 = __msa_srari_h(hz_out5, 5);

            res7 = __msa_srari_h(hz_out6, 5);

        } else {

            res1 = __msa_srari_h(hz_out2, 5);

            res3 = __msa_srari_h(hz_out3, 5);

            res5 = __msa_srari_h(hz_out4, 5);

            res7 = __msa_srari_h(hz_out5, 5);

        }



        SAT_SH4_SH(res1, res3, res5, res7, 7);



        res0 = __msa_aver_s_h(res0, res1);

        res1 = __msa_aver_s_h(res2, res3);

        res2 = __msa_aver_s_h(res4, res5);

        res3 = __msa_aver_s_h(res6, res7);

        ILVR_D2_UB(dst1, dst0, dst3, dst2, dst0, dst1);

        CONVERT_UB_AVG_ST8x4_UB(res0, res1, res2, res3, dst0, dst1,

                                dst, dst_stride);

        dst += (4 * dst_stride);



        hz_out0 = hz_out4;

        hz_out1 = hz_out5;

        hz_out2 = hz_out6;

        hz_out3 = hz_out7;

        hz_out4 = hz_out8;

    }

}
