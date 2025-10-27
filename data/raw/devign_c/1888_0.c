static void avc_loopfilter_luma_inter_edge_ver_msa(uint8_t *data,

                                                   uint8_t bs0, uint8_t bs1,

                                                   uint8_t bs2, uint8_t bs3,

                                                   uint8_t tc0, uint8_t tc1,

                                                   uint8_t tc2, uint8_t tc3,

                                                   uint8_t alpha_in,

                                                   uint8_t beta_in,

                                                   uint32_t img_width)

{

    uint8_t *src;

    v16u8 beta, tmp_vec, bs = { 0 };

    v16u8 tc = { 0 };

    v16u8 is_less_than, is_less_than_beta;

    v16u8 p1, p0, q0, q1;

    v8i16 p0_r, q0_r, p1_r = { 0 };

    v8i16 q1_r = { 0 };

    v8i16 p0_l, q0_l, p1_l = { 0 };

    v8i16 q1_l = { 0 };

    v16u8 p3_org, p2_org, p1_org, p0_org, q0_org, q1_org, q2_org, q3_org;

    v8i16 p2_org_r, p1_org_r, p0_org_r, q0_org_r, q1_org_r, q2_org_r;

    v8i16 p2_org_l, p1_org_l, p0_org_l, q0_org_l, q1_org_l, q2_org_l;

    v8i16 tc_r, tc_l;

    v16i8 zero = { 0 };

    v16u8 is_bs_greater_than0;



    tmp_vec = (v16u8) __msa_fill_b(bs0);

    bs = (v16u8) __msa_insve_w((v4i32) bs, 0, (v4i32) tmp_vec);

    tmp_vec = (v16u8) __msa_fill_b(bs1);

    bs = (v16u8) __msa_insve_w((v4i32) bs, 1, (v4i32) tmp_vec);

    tmp_vec = (v16u8) __msa_fill_b(bs2);

    bs = (v16u8) __msa_insve_w((v4i32) bs, 2, (v4i32) tmp_vec);

    tmp_vec = (v16u8) __msa_fill_b(bs3);

    bs = (v16u8) __msa_insve_w((v4i32) bs, 3, (v4i32) tmp_vec);



    if (!__msa_test_bz_v(bs)) {

        tmp_vec = (v16u8) __msa_fill_b(tc0);

        tc = (v16u8) __msa_insve_w((v4i32) tc, 0, (v4i32) tmp_vec);

        tmp_vec = (v16u8) __msa_fill_b(tc1);

        tc = (v16u8) __msa_insve_w((v4i32) tc, 1, (v4i32) tmp_vec);

        tmp_vec = (v16u8) __msa_fill_b(tc2);

        tc = (v16u8) __msa_insve_w((v4i32) tc, 2, (v4i32) tmp_vec);

        tmp_vec = (v16u8) __msa_fill_b(tc3);

        tc = (v16u8) __msa_insve_w((v4i32) tc, 3, (v4i32) tmp_vec);



        is_bs_greater_than0 = (zero < bs);



        {

            v16u8 row0, row1, row2, row3, row4, row5, row6, row7;

            v16u8 row8, row9, row10, row11, row12, row13, row14, row15;



            src = data;

            src -= 4;



            LOAD_8VECS_UB(src, img_width,

                          row0, row1, row2, row3, row4, row5, row6, row7);

            src += (8 * img_width);

            LOAD_8VECS_UB(src, img_width,

                          row8, row9, row10, row11, row12, row13, row14, row15);



            TRANSPOSE16x8_B_UB(row0, row1, row2, row3, row4, row5, row6, row7,

                               row8, row9, row10, row11,

                               row12, row13, row14, row15,

                               p3_org, p2_org, p1_org, p0_org,

                               q0_org, q1_org, q2_org, q3_org);

        }



        {

            v16u8 p0_asub_q0, p1_asub_p0, q1_asub_q0, alpha;

            v16u8 is_less_than_alpha;



            p0_asub_q0 = __msa_asub_u_b(p0_org, q0_org);

            p1_asub_p0 = __msa_asub_u_b(p1_org, p0_org);

            q1_asub_q0 = __msa_asub_u_b(q1_org, q0_org);



            alpha = (v16u8) __msa_fill_b(alpha_in);

            beta = (v16u8) __msa_fill_b(beta_in);



            is_less_than_alpha = (p0_asub_q0 < alpha);

            is_less_than_beta = (p1_asub_p0 < beta);

            is_less_than = is_less_than_beta & is_less_than_alpha;

            is_less_than_beta = (q1_asub_q0 < beta);

            is_less_than = is_less_than_beta & is_less_than;

            is_less_than = is_less_than & is_bs_greater_than0;

        }



        if (!__msa_test_bz_v(is_less_than)) {

            v16i8 negate_tc, sign_negate_tc;

            v8i16 negate_tc_r, i16_negatetc_l;



            negate_tc = zero - (v16i8) tc;

            sign_negate_tc = __msa_clti_s_b(negate_tc, 0);



            negate_tc_r = (v8i16) __msa_ilvr_b(sign_negate_tc, negate_tc);

            i16_negatetc_l = (v8i16) __msa_ilvl_b(sign_negate_tc, negate_tc);



            tc_r = (v8i16) __msa_ilvr_b(zero, (v16i8) tc);

            tc_l = (v8i16) __msa_ilvl_b(zero, (v16i8) tc);



            p1_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) p1_org);

            p0_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) p0_org);

            q0_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) q0_org);



            p1_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) p1_org);

            p0_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) p0_org);

            q0_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) q0_org);



            {

                v16u8 p2_asub_p0;

                v16u8 is_less_than_beta_r, is_less_than_beta_l;



                p2_asub_p0 = __msa_asub_u_b(p2_org, p0_org);

                is_less_than_beta = (p2_asub_p0 < beta);

                is_less_than_beta = is_less_than_beta & is_less_than;



                is_less_than_beta_r =

                    (v16u8) __msa_sldi_b((v16i8) is_less_than_beta, zero, 8);

                if (!__msa_test_bz_v(is_less_than_beta_r)) {

                    p2_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) p2_org);



                    AVC_LOOP_FILTER_P1_OR_Q1(p0_org_r, q0_org_r,

                                             p1_org_r, p2_org_r,

                                             negate_tc_r, tc_r, p1_r);

                }



                is_less_than_beta_l =

                    (v16u8) __msa_sldi_b(zero, (v16i8) is_less_than_beta, 8);

                if (!__msa_test_bz_v(is_less_than_beta_l)) {

                    p2_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) p2_org);



                    AVC_LOOP_FILTER_P1_OR_Q1(p0_org_l, q0_org_l,

                                             p1_org_l, p2_org_l,

                                             i16_negatetc_l, tc_l, p1_l);

                }

            }



            if (!__msa_test_bz_v(is_less_than_beta)) {

                p1 = (v16u8) __msa_pckev_b((v16i8) p1_l, (v16i8) p1_r);

                p1_org = __msa_bmnz_v(p1_org, p1, is_less_than_beta);



                is_less_than_beta = __msa_andi_b(is_less_than_beta, 1);

                tc = tc + is_less_than_beta;

            }



            {

                v16u8 u8_q2asub_q0;

                v16u8 is_less_than_beta_l, is_less_than_beta_r;



                u8_q2asub_q0 = __msa_asub_u_b(q2_org, q0_org);

                is_less_than_beta = (u8_q2asub_q0 < beta);

                is_less_than_beta = is_less_than_beta & is_less_than;



                q1_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) q1_org);



                is_less_than_beta_r =

                    (v16u8) __msa_sldi_b((v16i8) is_less_than_beta, zero, 8);

                if (!__msa_test_bz_v(is_less_than_beta_r)) {

                    q2_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) q2_org);



                    AVC_LOOP_FILTER_P1_OR_Q1(p0_org_r, q0_org_r,

                                             q1_org_r, q2_org_r,

                                             negate_tc_r, tc_r, q1_r);

                }



                q1_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) q1_org);



                is_less_than_beta_l =

                    (v16u8) __msa_sldi_b(zero, (v16i8) is_less_than_beta, 8);

                if (!__msa_test_bz_v(is_less_than_beta_l)) {

                    q2_org_l = (v8i16) __msa_ilvl_b(zero, (v16i8) q2_org);



                    AVC_LOOP_FILTER_P1_OR_Q1(p0_org_l, q0_org_l,

                                             q1_org_l, q2_org_l,

                                             i16_negatetc_l, tc_l, q1_l);

                }

            }



            if (!__msa_test_bz_v(is_less_than_beta)) {

                q1 = (v16u8) __msa_pckev_b((v16i8) q1_l, (v16i8) q1_r);

                q1_org = __msa_bmnz_v(q1_org, q1, is_less_than_beta);



                is_less_than_beta = __msa_andi_b(is_less_than_beta, 1);

                tc = tc + is_less_than_beta;

            }



            {

                v8i16 threshold_r, negate_thresh_r;

                v8i16 threshold_l, negate_thresh_l;

                v16i8 negate_thresh, sign_negate_thresh;



                negate_thresh = zero - (v16i8) tc;

                sign_negate_thresh = __msa_clti_s_b(negate_thresh, 0);



                threshold_r = (v8i16) __msa_ilvr_b(zero, (v16i8) tc);

                negate_thresh_r = (v8i16) __msa_ilvr_b(sign_negate_thresh,

                                                       negate_thresh);



                AVC_LOOP_FILTER_P0Q0(q0_org_r, p0_org_r, p1_org_r, q1_org_r,

                                     negate_thresh_r, threshold_r, p0_r, q0_r);



                threshold_l = (v8i16) __msa_ilvl_b(zero, (v16i8) tc);

                negate_thresh_l = (v8i16) __msa_ilvl_b(sign_negate_thresh,

                                                       negate_thresh);



                AVC_LOOP_FILTER_P0Q0(q0_org_l, p0_org_l, p1_org_l, q1_org_l,

                                     negate_thresh_l, threshold_l, p0_l, q0_l);

            }



            p0 = (v16u8) __msa_pckev_b((v16i8) p0_l, (v16i8) p0_r);

            q0 = (v16u8) __msa_pckev_b((v16i8) q0_l, (v16i8) q0_r);



            p0_org = __msa_bmnz_v(p0_org, p0, is_less_than);

            q0_org = __msa_bmnz_v(q0_org, q0, is_less_than);

        }



        {

            v16i8 tmp0, tmp1;

            v8i16 tmp2, tmp5;

            v4i32 tmp3, tmp4, tmp6, tmp7;

            uint32_t out0, out2;

            uint16_t out1, out3;



            src = data - 3;



            tmp0 = __msa_ilvr_b((v16i8) p1_org, (v16i8) p2_org);

            tmp1 = __msa_ilvr_b((v16i8) q0_org, (v16i8) p0_org);

            tmp2 = (v8i16) __msa_ilvr_b((v16i8) q2_org, (v16i8) q1_org);

            tmp3 = (v4i32) __msa_ilvr_h((v8i16) tmp1, (v8i16) tmp0);

            tmp4 = (v4i32) __msa_ilvl_h((v8i16) tmp1, (v8i16) tmp0);



            tmp0 = __msa_ilvl_b((v16i8) p1_org, (v16i8) p2_org);

            tmp1 = __msa_ilvl_b((v16i8) q0_org, (v16i8) p0_org);

            tmp5 = (v8i16) __msa_ilvl_b((v16i8) q2_org, (v16i8) q1_org);

            tmp6 = (v4i32) __msa_ilvr_h((v8i16) tmp1, (v8i16) tmp0);

            tmp7 = (v4i32) __msa_ilvl_h((v8i16) tmp1, (v8i16) tmp0);



            out0 = __msa_copy_u_w(tmp3, 0);

            out1 = __msa_copy_u_h(tmp2, 0);

            out2 = __msa_copy_u_w(tmp3, 1);

            out3 = __msa_copy_u_h(tmp2, 1);



            STORE_WORD(src, out0);

            STORE_HWORD((src + 4), out1);

            src += img_width;

            STORE_WORD(src, out2);

            STORE_HWORD((src + 4), out3);



            out0 = __msa_copy_u_w(tmp3, 2);

            out1 = __msa_copy_u_h(tmp2, 2);

            out2 = __msa_copy_u_w(tmp3, 3);

            out3 = __msa_copy_u_h(tmp2, 3);



            src += img_width;

            STORE_WORD(src, out0);

            STORE_HWORD((src + 4), out1);

            src += img_width;

            STORE_WORD(src, out2);

            STORE_HWORD((src + 4), out3);



            out0 = __msa_copy_u_w(tmp4, 0);

            out1 = __msa_copy_u_h(tmp2, 4);

            out2 = __msa_copy_u_w(tmp4, 1);

            out3 = __msa_copy_u_h(tmp2, 5);



            src += img_width;

            STORE_WORD(src, out0);

            STORE_HWORD((src + 4), out1);

            src += img_width;

            STORE_WORD(src, out2);

            STORE_HWORD((src + 4), out3);



            out0 = __msa_copy_u_w(tmp4, 2);

            out1 = __msa_copy_u_h(tmp2, 6);

            out2 = __msa_copy_u_w(tmp4, 3);

            out3 = __msa_copy_u_h(tmp2, 7);



            src += img_width;

            STORE_WORD(src, out0);

            STORE_HWORD((src + 4), out1);

            src += img_width;

            STORE_WORD(src, out2);

            STORE_HWORD((src + 4), out3);



            out0 = __msa_copy_u_w(tmp6, 0);

            out1 = __msa_copy_u_h(tmp5, 0);

            out2 = __msa_copy_u_w(tmp6, 1);

            out3 = __msa_copy_u_h(tmp5, 1);



            src += img_width;

            STORE_WORD(src, out0);

            STORE_HWORD((src + 4), out1);

            src += img_width;

            STORE_WORD(src, out2);

            STORE_HWORD((src + 4), out3);



            out0 = __msa_copy_u_w(tmp6, 2);

            out1 = __msa_copy_u_h(tmp5, 2);

            out2 = __msa_copy_u_w(tmp6, 3);

            out3 = __msa_copy_u_h(tmp5, 3);



            src += img_width;

            STORE_WORD(src, out0);

            STORE_HWORD((src + 4), out1);

            src += img_width;

            STORE_WORD(src, out2);

            STORE_HWORD((src + 4), out3);



            out0 = __msa_copy_u_w(tmp7, 0);

            out1 = __msa_copy_u_h(tmp5, 4);

            out2 = __msa_copy_u_w(tmp7, 1);

            out3 = __msa_copy_u_h(tmp5, 5);



            src += img_width;

            STORE_WORD(src, out0);

            STORE_HWORD((src + 4), out1);

            src += img_width;

            STORE_WORD(src, out2);

            STORE_HWORD((src + 4), out3);



            out0 = __msa_copy_u_w(tmp7, 2);

            out1 = __msa_copy_u_h(tmp5, 6);

            out2 = __msa_copy_u_w(tmp7, 3);

            out3 = __msa_copy_u_h(tmp5, 7);



            src += img_width;

            STORE_WORD(src, out0);

            STORE_HWORD((src + 4), out1);

            src += img_width;

            STORE_WORD(src, out2);

            STORE_HWORD((src + 4), out3);

        }

    }

}
