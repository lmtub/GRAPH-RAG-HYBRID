static void avc_h_loop_filter_chroma422_msa(uint8_t *src,

                                            int32_t stride,

                                            int32_t alpha_in,

                                            int32_t beta_in,

                                            int8_t *tc0)

{

    int32_t col, tc_val;

    int16_t out0, out1, out2, out3;

    v16u8 alpha, beta, res;



    alpha = (v16u8) __msa_fill_b(alpha_in);

    beta = (v16u8) __msa_fill_b(beta_in);



    for (col = 0; col < 4; col++) {

        tc_val = (tc0[col] - 1) + 1;



        if (tc_val <= 0) {

            src += (4 * stride);

            continue;

        }



        AVC_LPF_H_CHROMA_422(src, stride, tc_val, alpha, beta, res);



        out0 = __msa_copy_s_h((v8i16) res, 0);

        out1 = __msa_copy_s_h((v8i16) res, 1);

        out2 = __msa_copy_s_h((v8i16) res, 2);

        out3 = __msa_copy_s_h((v8i16) res, 3);



        STORE_HWORD((src - 1), out0);

        src += stride;

        STORE_HWORD((src - 1), out1);

        src += stride;

        STORE_HWORD((src - 1), out2);

        src += stride;

        STORE_HWORD((src - 1), out3);

        src += stride;

    }

}
