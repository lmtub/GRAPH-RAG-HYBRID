static void intra_predict_vert_dc_8x8_msa(uint8_t *src, int32_t stride)

{

    uint8_t lp_cnt;

    uint32_t out0 = 0, out1 = 0;

    v16u8 src_top;

    v8u16 add;

    v4u32 sum;

    v4i32 res0, res1;



    src_top = LD_UB(src - stride);

    add = __msa_hadd_u_h(src_top, src_top);

    sum = __msa_hadd_u_w(add, add);

    sum = (v4u32) __msa_srari_w((v4i32) sum, 2);

    res0 = (v4i32) __msa_splati_b((v16i8) sum, 0);

    res1 = (v4i32) __msa_splati_b((v16i8) sum, 4);

    out0 = __msa_copy_u_w(res0, 0);

    out1 = __msa_copy_u_w(res1, 0);



    for (lp_cnt = 8; lp_cnt--;) {

        SW(out0, src);

        SW(out1, src + 4);

        src += stride;

    }

}
