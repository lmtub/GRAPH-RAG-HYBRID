static void avc_luma_midh_qrt_4w_msa(const uint8_t *src, int32_t src_stride,

                                     uint8_t *dst, int32_t dst_stride,

                                     int32_t height, uint8_t horiz_offset)

{

    uint32_t row;

    v16i8 src0, src1, src2, src3, src4, src5, src6;

    v8i16 vt_res0, vt_res1, vt_res2, vt_res3;

    v4i32 hz_res0, hz_res1;

    v8i16 dst0, dst1;

    v8i16 shf_vec0, shf_vec1, shf_vec2, shf_vec3, shf_vec4, shf_vec5;

    v8i16 mask0 = { 0, 5, 1, 6, 2, 7, 3, 8 };

    v8i16 mask1 = { 1, 4, 2, 5, 3, 6, 4, 7 };

    v8i16 mask2 = { 2, 3, 3, 4, 4, 5, 5, 6 };

    v8i16 minus5h = __msa_ldi_h(-5);

    v8i16 plus20h = __msa_ldi_h(20);

    v8i16 zeros = { 0 };

    v16u8 out;



    LD_SB5(src, src_stride, src0, src1, src2, src3, src4);

    src += (5 * src_stride);

    XORI_B5_128_SB(src0, src1, src2, src3, src4);



    for (row = (height >> 1); row--;) {

        LD_SB2(src, src_stride, src5, src6);

        src += (2 * src_stride);



        XORI_B2_128_SB(src5, src6);

        AVC_CALC_DPADD_B_6PIX_2COEFF_SH(src0, src1, src2, src3, src4, src5,

                                        vt_res0, vt_res1);

        AVC_CALC_DPADD_B_6PIX_2COEFF_SH(src1, src2, src3, src4, src5, src6,

                                        vt_res2, vt_res3);

        VSHF_H3_SH(vt_res0, vt_res1, vt_res0, vt_res1, vt_res0, vt_res1,

                   mask0, mask1, mask2, shf_vec0, shf_vec1, shf_vec2);

        VSHF_H3_SH(vt_res2, vt_res3, vt_res2, vt_res3, vt_res2, vt_res3,

                   mask0, mask1, mask2, shf_vec3, shf_vec4, shf_vec5);

        hz_res0 = __msa_hadd_s_w(shf_vec0, shf_vec0);

        DPADD_SH2_SW(shf_vec1, shf_vec2, minus5h, plus20h, hz_res0, hz_res0);

        hz_res1 = __msa_hadd_s_w(shf_vec3, shf_vec3);

        DPADD_SH2_SW(shf_vec4, shf_vec5, minus5h, plus20h, hz_res1, hz_res1);



        SRARI_W2_SW(hz_res0, hz_res1, 10);

        SAT_SW2_SW(hz_res0, hz_res1, 7);



        dst0 = __msa_srari_h(shf_vec2, 5);

        dst1 = __msa_srari_h(shf_vec5, 5);



        SAT_SH2_SH(dst0, dst1, 7);



        if (horiz_offset) {

            dst0 = __msa_ilvod_h(zeros, dst0);

            dst1 = __msa_ilvod_h(zeros, dst1);

        } else {

            ILVEV_H2_SH(dst0, zeros, dst1, zeros, dst0, dst1);

        }



        hz_res0 = __msa_aver_s_w(hz_res0, (v4i32) dst0);

        hz_res1 = __msa_aver_s_w(hz_res1, (v4i32) dst1);

        dst0 = __msa_pckev_h((v8i16) hz_res1, (v8i16) hz_res0);



        out = PCKEV_XORI128_UB(dst0, dst0);

        ST4x2_UB(out, dst, dst_stride);



        dst += (2 * dst_stride);



        src0 = src2;

        src1 = src3;

        src2 = src4;

        src3 = src5;

        src4 = src6;

    }

}
