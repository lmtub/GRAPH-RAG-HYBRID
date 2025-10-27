static void avc_luma_midv_qrt_and_aver_dst_4w_msa(const uint8_t *src,

                                                  int32_t src_stride,

                                                  uint8_t *dst,

                                                  int32_t dst_stride,

                                                  int32_t height,

                                                  uint8_t ver_offset)

{

    int32_t loop_cnt;

    int32_t out0, out1;

    v16i8 src0, src1, src2, src3, src4;

    v16u8 dst0, dst1;

    v16i8 mask0, mask1, mask2;

    v8i16 hz_out0, hz_out1, hz_out2, hz_out3;

    v8i16 hz_out4, hz_out5, hz_out6;

    v8i16 res0, res1, res2, res3;

    v16u8 vec0, vec1;



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



    for (loop_cnt = (height >> 1); loop_cnt--;) {

        LD_SB2(src, src_stride, src0, src1);

        src += (2 * src_stride);



        XORI_B2_128_SB(src0, src1);

        LD_UB2(dst, dst_stride, dst0, dst1);

        hz_out5 = AVC_XOR_VSHF_B_AND_APPLY_6TAP_HORIZ_FILT_SH(src0, src1,

                                                              mask0, mask1,

                                                              mask2);

        hz_out6 = (v8i16) __msa_pckod_d((v2i64) hz_out5, (v2i64) hz_out5);

        res0 = AVC_CALC_DPADD_H_6PIX_2COEFF_R_SH(hz_out0, hz_out1, hz_out2,

                                                 hz_out3, hz_out4, hz_out5);

        res2 = AVC_CALC_DPADD_H_6PIX_2COEFF_R_SH(hz_out1, hz_out2, hz_out3,

                                                 hz_out4, hz_out5, hz_out6);



        if (ver_offset) {

            res1 = __msa_srari_h(hz_out3, 5);

            res3 = __msa_srari_h(hz_out4, 5);

        } else {

            res1 = __msa_srari_h(hz_out2, 5);

            res3 = __msa_srari_h(hz_out3, 5);

        }



        SAT_SH2_SH(res1, res3, 7);



        res0 = __msa_aver_s_h(res0, res1);

        res1 = __msa_aver_s_h(res2, res3);



        vec0 = PCKEV_XORI128_UB(res0, res0);

        vec1 = PCKEV_XORI128_UB(res1, res1);



        AVER_UB2_UB(vec0, dst0, vec1, dst1, dst0, dst1);



        out0 = __msa_copy_u_w((v4i32) dst0, 0);

        out1 = __msa_copy_u_w((v4i32) dst1, 0);

        SW(out0, dst);

        dst += dst_stride;

        SW(out1, dst);

        dst += dst_stride;



        hz_out0 = hz_out2;

        hz_out1 = hz_out3;

        hz_out2 = hz_out4;

        hz_out3 = hz_out5;

        hz_out4 = hz_out6;

    }

}
