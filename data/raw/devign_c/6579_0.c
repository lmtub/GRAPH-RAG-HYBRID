static void avc_luma_midv_qrt_8w_msa(const uint8_t *src, int32_t src_stride,

                                     uint8_t *dst, int32_t dst_stride,

                                     int32_t height, uint8_t ver_offset)

{

    uint32_t loop_cnt;

    v16i8 src0, src1, src2, src3, src4;

    v16i8 mask0, mask1, mask2;

    v8i16 hz_out0, hz_out1, hz_out2, hz_out3;

    v8i16 hz_out4, hz_out5, hz_out6, hz_out7, hz_out8;

    v8i16 dst0, dst1, dst2, dst3, dst4, dst5, dst6, dst7;

    v16u8 out;



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



        hz_out5 = AVC_HORZ_FILTER_SH(src0, mask0, mask1, mask2);

        hz_out6 = AVC_HORZ_FILTER_SH(src1, mask0, mask1, mask2);

        hz_out7 = AVC_HORZ_FILTER_SH(src2, mask0, mask1, mask2);

        hz_out8 = AVC_HORZ_FILTER_SH(src3, mask0, mask1, mask2);



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



        out = PCKEV_XORI128_UB(dst0, dst0);

        ST8x1_UB(out, dst);

        dst += dst_stride;

        out = PCKEV_XORI128_UB(dst1, dst1);

        ST8x1_UB(out, dst);

        dst += dst_stride;

        out = PCKEV_XORI128_UB(dst2, dst2);

        ST8x1_UB(out, dst);

        dst += dst_stride;

        out = PCKEV_XORI128_UB(dst3, dst3);

        ST8x1_UB(out, dst);

        dst += dst_stride;



        hz_out0 = hz_out4;

        hz_out1 = hz_out5;

        hz_out2 = hz_out6;

        hz_out3 = hz_out7;

        hz_out4 = hz_out8;

    }

}
