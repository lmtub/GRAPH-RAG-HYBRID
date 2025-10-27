static void avc_luma_hv_qrt_and_aver_dst_4x4_msa(const uint8_t *src_x,

                                                 const uint8_t *src_y,

                                                 int32_t src_stride,

                                                 uint8_t *dst,

                                                 int32_t dst_stride)

{

    v16i8 src_hz0, src_hz1, src_hz2, src_hz3;

    v16u8 dst0, dst1, dst2, dst3;

    v16i8 src_vt0, src_vt1, src_vt2, src_vt3, src_vt4;

    v16i8 src_vt5, src_vt6, src_vt7, src_vt8;

    v16i8 mask0, mask1, mask2;

    v8i16 hz_out0, hz_out1, vert_out0, vert_out1;

    v8i16 res0, res1;

    v16u8 res;



    LD_SB3(&luma_mask_arr[48], 16, mask0, mask1, mask2);

    LD_SB5(src_y, src_stride, src_vt0, src_vt1, src_vt2, src_vt3, src_vt4);

    src_y += (5 * src_stride);



    src_vt0 = (v16i8) __msa_insve_w((v4i32) src_vt0, 1, (v4i32) src_vt1);

    src_vt1 = (v16i8) __msa_insve_w((v4i32) src_vt1, 1, (v4i32) src_vt2);

    src_vt2 = (v16i8) __msa_insve_w((v4i32) src_vt2, 1, (v4i32) src_vt3);

    src_vt3 = (v16i8) __msa_insve_w((v4i32) src_vt3, 1, (v4i32) src_vt4);



    XORI_B4_128_SB(src_vt0, src_vt1, src_vt2, src_vt3);

    LD_SB4(src_x, src_stride, src_hz0, src_hz1, src_hz2, src_hz3);

    LD_UB4(dst, dst_stride, dst0, dst1, dst2, dst3);

    XORI_B4_128_SB(src_hz0, src_hz1, src_hz2, src_hz3);

    hz_out0 = AVC_XOR_VSHF_B_AND_APPLY_6TAP_HORIZ_FILT_SH(src_hz0, src_hz1,

                                                          mask0, mask1, mask2);

    hz_out1 = AVC_XOR_VSHF_B_AND_APPLY_6TAP_HORIZ_FILT_SH(src_hz2, src_hz3,

                                                          mask0, mask1, mask2);

    SRARI_H2_SH(hz_out0, hz_out1, 5);

    SAT_SH2_SH(hz_out0, hz_out1, 7);

    LD_SB4(src_y, src_stride, src_vt5, src_vt6, src_vt7, src_vt8);



    src_vt4 = (v16i8) __msa_insve_w((v4i32) src_vt4, 1, (v4i32) src_vt5);

    src_vt5 = (v16i8) __msa_insve_w((v4i32) src_vt5, 1, (v4i32) src_vt6);

    src_vt6 = (v16i8) __msa_insve_w((v4i32) src_vt6, 1, (v4i32) src_vt7);

    src_vt7 = (v16i8) __msa_insve_w((v4i32) src_vt7, 1, (v4i32) src_vt8);



    XORI_B4_128_SB(src_vt4, src_vt5, src_vt6, src_vt7);



    /* filter calc */

    vert_out0 = AVC_CALC_DPADD_B_6PIX_2COEFF_R_SH(src_vt0, src_vt1, src_vt2,

                                                  src_vt3, src_vt4, src_vt5);

    vert_out1 = AVC_CALC_DPADD_B_6PIX_2COEFF_R_SH(src_vt2, src_vt3, src_vt4,

                                                  src_vt5, src_vt6, src_vt7);

    SRARI_H2_SH(vert_out0, vert_out1, 5);

    SAT_SH2_SH(vert_out0, vert_out1, 7);



    res1 = __msa_srari_h((hz_out1 + vert_out1), 1);

    res0 = __msa_srari_h((hz_out0 + vert_out0), 1);



    SAT_SH2_SH(res0, res1, 7);

    res = PCKEV_XORI128_UB(res0, res1);



    dst0 = (v16u8) __msa_insve_w((v4i32) dst0, 1, (v4i32) dst1);

    dst1 = (v16u8) __msa_insve_w((v4i32) dst2, 1, (v4i32) dst3);

    dst0 = (v16u8) __msa_insve_d((v2i64) dst0, 1, (v2i64) dst1);

    dst0 = __msa_aver_u_b(res, dst0);



    ST4x4_UB(dst0, dst0, 0, 1, 2, 3, dst, dst_stride);

}
