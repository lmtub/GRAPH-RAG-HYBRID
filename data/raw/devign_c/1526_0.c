static void avc_luma_hv_qrt_and_aver_dst_8x8_msa(const uint8_t *src_x,

                                                 const uint8_t *src_y,

                                                 int32_t src_stride,

                                                 uint8_t *dst,

                                                 int32_t dst_stride)

{

    uint32_t loop_cnt;

    v16i8 src_hz0, src_hz1, src_hz2, src_hz3;

    v16u8 dst0, dst1, dst2, dst3;

    v16i8 src_vt0, src_vt1, src_vt2, src_vt3;

    v16i8 src_vt4, src_vt5, src_vt6, src_vt7, src_vt8;

    v16i8 mask0, mask1, mask2;

    v8i16 hz_out0, hz_out1, hz_out2, hz_out3;

    v8i16 vert_out0, vert_out1, vert_out2, vert_out3;

    v8i16 out0, out1, out2, out3;



    LD_SB3(&luma_mask_arr[0], 16, mask0, mask1, mask2);



    LD_SB5(src_y, src_stride, src_vt0, src_vt1, src_vt2, src_vt3, src_vt4);

    src_y += (5 * src_stride);



    src_vt0 = (v16i8) __msa_insve_d((v2i64) src_vt0, 1, (v2i64) src_vt1);

    src_vt1 = (v16i8) __msa_insve_d((v2i64) src_vt1, 1, (v2i64) src_vt2);

    src_vt2 = (v16i8) __msa_insve_d((v2i64) src_vt2, 1, (v2i64) src_vt3);

    src_vt3 = (v16i8) __msa_insve_d((v2i64) src_vt3, 1, (v2i64) src_vt4);



    XORI_B4_128_SB(src_vt0, src_vt1, src_vt2, src_vt3);



    for (loop_cnt = 2; loop_cnt--;) {

        LD_SB4(src_x, src_stride, src_hz0, src_hz1, src_hz2, src_hz3);

        XORI_B4_128_SB(src_hz0, src_hz1, src_hz2, src_hz3);

        src_x += (4 * src_stride);



        LD_UB4(dst, dst_stride, dst0, dst1, dst2, dst3);

        hz_out0 = AVC_HORZ_FILTER_SH(src_hz0, mask0, mask1, mask2);

        hz_out1 = AVC_HORZ_FILTER_SH(src_hz1, mask0, mask1, mask2);

        hz_out2 = AVC_HORZ_FILTER_SH(src_hz2, mask0, mask1, mask2);

        hz_out3 = AVC_HORZ_FILTER_SH(src_hz3, mask0, mask1, mask2);

        SRARI_H4_SH(hz_out0, hz_out1, hz_out2, hz_out3, 5);

        SAT_SH4_SH(hz_out0, hz_out1, hz_out2, hz_out3, 7);

        LD_SB4(src_y, src_stride, src_vt5, src_vt6, src_vt7, src_vt8);

        src_y += (4 * src_stride);



        src_vt4 = (v16i8) __msa_insve_d((v2i64) src_vt4, 1, (v2i64) src_vt5);

        src_vt5 = (v16i8) __msa_insve_d((v2i64) src_vt5, 1, (v2i64) src_vt6);

        src_vt6 = (v16i8) __msa_insve_d((v2i64) src_vt6, 1, (v2i64) src_vt7);

        src_vt7 = (v16i8) __msa_insve_d((v2i64) src_vt7, 1, (v2i64) src_vt8);



        XORI_B4_128_SB(src_vt4, src_vt5, src_vt6, src_vt7);

        AVC_CALC_DPADD_B_6PIX_2COEFF_SH(src_vt0, src_vt1, src_vt2, src_vt3,

                                        src_vt4, src_vt5, vert_out0, vert_out1);

        AVC_CALC_DPADD_B_6PIX_2COEFF_SH(src_vt2, src_vt3, src_vt4, src_vt5,

                                        src_vt6, src_vt7, vert_out2, vert_out3);

        SRARI_H4_SH(vert_out0, vert_out1, vert_out2, vert_out3, 5);

        SAT_SH4_SH(vert_out0, vert_out1, vert_out2, vert_out3, 7);



        out0 = __msa_srari_h((hz_out0 + vert_out0), 1);

        out1 = __msa_srari_h((hz_out1 + vert_out1), 1);

        out2 = __msa_srari_h((hz_out2 + vert_out2), 1);

        out3 = __msa_srari_h((hz_out3 + vert_out3), 1);



        SAT_SH4_SH(out0, out1, out2, out3, 7);

        ILVR_D2_UB(dst1, dst0, dst3, dst2, dst0, dst1);

        CONVERT_UB_AVG_ST8x4_UB(out0, out1, out2, out3, dst0, dst1,

                                dst, dst_stride);

        dst += (4 * dst_stride);



        src_vt0 = src_vt4;

        src_vt1 = src_vt5;

        src_vt2 = src_vt6;

        src_vt3 = src_vt7;

        src_vt4 = src_vt8;

    }

}
