static void avc_loopfilter_cb_or_cr_intra_edge_ver_msa(uint8_t *data_cb_or_cr,

                                                       uint8_t alpha_in,

                                                       uint8_t beta_in,

                                                       uint32_t img_width)

{

    uint16_t out0, out1, out2, out3;

    v8i16 tmp1;

    v16u8 alpha, beta, is_less_than;

    v8i16 p0_or_q0, q0_or_p0;

    v16u8 p1_or_q1_org, p0_or_q0_org, q0_or_p0_org, q1_or_p1_org;

    v16i8 zero = { 0 };

    v16u8 p0_asub_q0, p1_asub_p0, q1_asub_q0;

    v16u8 is_less_than_alpha, is_less_than_beta;

    v8i16 p1_org_r, p0_org_r, q0_org_r, q1_org_r;



    {

        v16u8 row0, row1, row2, row3, row4, row5, row6, row7;



        LOAD_8VECS_UB((data_cb_or_cr - 2), img_width,

                      row0, row1, row2, row3, row4, row5, row6, row7);



        TRANSPOSE8x4_B_UB(row0, row1, row2, row3, row4, row5, row6, row7,

                          p1_or_q1_org, p0_or_q0_org,

                          q0_or_p0_org, q1_or_p1_org);

    }



    alpha = (v16u8) __msa_fill_b(alpha_in);

    beta = (v16u8) __msa_fill_b(beta_in);



    p0_asub_q0 = __msa_asub_u_b(p0_or_q0_org, q0_or_p0_org);

    p1_asub_p0 = __msa_asub_u_b(p1_or_q1_org, p0_or_q0_org);

    q1_asub_q0 = __msa_asub_u_b(q1_or_p1_org, q0_or_p0_org);



    is_less_than_alpha = (p0_asub_q0 < alpha);

    is_less_than_beta = (p1_asub_p0 < beta);

    is_less_than = is_less_than_beta & is_less_than_alpha;

    is_less_than_beta = (q1_asub_q0 < beta);

    is_less_than = is_less_than_beta & is_less_than;



    is_less_than = (v16u8) __msa_ilvr_d((v2i64) zero, (v2i64) is_less_than);



    if (!__msa_test_bz_v(is_less_than)) {

        p1_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) p1_or_q1_org);

        p0_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) p0_or_q0_org);

        q0_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) q0_or_p0_org);

        q1_org_r = (v8i16) __msa_ilvr_b(zero, (v16i8) q1_or_p1_org);



        AVC_LOOP_FILTER_P0_OR_Q0(p0_org_r, q1_org_r, p1_org_r, p0_or_q0);

        AVC_LOOP_FILTER_P0_OR_Q0(q0_org_r, p1_org_r, q1_org_r, q0_or_p0);



        /* convert 16 bit output into 8 bit output */

        p0_or_q0 = (v8i16) __msa_pckev_b(zero, (v16i8) p0_or_q0);

        q0_or_p0 = (v8i16) __msa_pckev_b(zero, (v16i8) q0_or_p0);



        p0_or_q0_org =

            __msa_bmnz_v(p0_or_q0_org, (v16u8) p0_or_q0, is_less_than);

        q0_or_p0_org =

            __msa_bmnz_v(q0_or_p0_org, (v16u8) q0_or_p0, is_less_than);



        tmp1 = (v8i16) __msa_ilvr_b((v16i8) q0_or_p0_org, (v16i8) p0_or_q0_org);



        data_cb_or_cr -= 1;



        out0 = __msa_copy_u_h(tmp1, 0);

        out1 = __msa_copy_u_h(tmp1, 1);

        out2 = __msa_copy_u_h(tmp1, 2);

        out3 = __msa_copy_u_h(tmp1, 3);



        STORE_HWORD(data_cb_or_cr, out0);

        data_cb_or_cr += img_width;

        STORE_HWORD(data_cb_or_cr, out1);

        data_cb_or_cr += img_width;

        STORE_HWORD(data_cb_or_cr, out2);

        data_cb_or_cr += img_width;

        STORE_HWORD(data_cb_or_cr, out3);

        data_cb_or_cr += img_width;



        out0 = __msa_copy_u_h(tmp1, 4);

        out1 = __msa_copy_u_h(tmp1, 5);

        out2 = __msa_copy_u_h(tmp1, 6);

        out3 = __msa_copy_u_h(tmp1, 7);



        STORE_HWORD(data_cb_or_cr, out0);

        data_cb_or_cr += img_width;

        STORE_HWORD(data_cb_or_cr, out1);

        data_cb_or_cr += img_width;

        STORE_HWORD(data_cb_or_cr, out2);

        data_cb_or_cr += img_width;

        STORE_HWORD(data_cb_or_cr, out3);

    }

}
