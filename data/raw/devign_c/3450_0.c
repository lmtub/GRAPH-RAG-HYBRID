static void intra_predict_plane_16x16_msa(uint8_t *src, int32_t stride)

{

    uint8_t lpcnt;

    int32_t res0, res1, res2, res3;

    uint64_t load0, load1;

    v16i8 shf_mask = { 7, 8, 6, 9, 5, 10, 4, 11, 3, 12, 2, 13, 1, 14, 0, 15 };

    v8i16 short_multiplier = { 1, 2, 3, 4, 5, 6, 7, 8 };

    v4i32 int_multiplier = { 0, 1, 2, 3 };

    v16u8 src_top = { 0 };

    v8i16 vec9, vec10;

    v4i32 vec0, vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, res_add;



    load0 = LD(src - (stride + 1));

    load1 = LD(src - (stride + 1) + 9);



    INSERT_D2_UB(load0, load1, src_top);



    src_top = (v16u8) __msa_vshf_b(shf_mask, (v16i8) src_top, (v16i8) src_top);



    vec9 = __msa_hsub_u_h(src_top, src_top);

    vec9 *= short_multiplier;

    vec8 = __msa_hadd_s_w(vec9, vec9);

    res_add = (v4i32) __msa_hadd_s_d(vec8, vec8);



    res0 = __msa_copy_s_w(res_add, 0) + __msa_copy_s_w(res_add, 2);



    res1 = (src[8 * stride - 1] - src[6 * stride - 1]) +

        2 * (src[9 * stride - 1] - src[5 * stride - 1]) +

        3 * (src[10 * stride - 1] - src[4 * stride - 1]) +

        4 * (src[11 * stride - 1] - src[3 * stride - 1]) +

        5 * (src[12 * stride - 1] - src[2 * stride - 1]) +

        6 * (src[13 * stride - 1] - src[stride - 1]) +

        7 * (src[14 * stride - 1] - src[-1]) +

        8 * (src[15 * stride - 1] - src[-1 * stride - 1]);



    res0 *= 5;

    res1 *= 5;

    res0 = (res0 + 32) >> 6;

    res1 = (res1 + 32) >> 6;



    res3 = 7 * (res0 + res1);

    res2 = 16 * (src[15 * stride - 1] + src[-stride + 15] + 1);

    res2 -= res3;



    vec8 = __msa_fill_w(res0);

    vec4 = __msa_fill_w(res2);

    vec5 = __msa_fill_w(res1);

    vec6 = vec8 * 4;

    vec7 = vec8 * int_multiplier;



    for (lpcnt = 16; lpcnt--;) {

        vec0 = vec7;

        vec0 += vec4;

        vec1 = vec0 + vec6;

        vec2 = vec1 + vec6;

        vec3 = vec2 + vec6;



        SRA_4V(vec0, vec1, vec2, vec3, 5);

        PCKEV_H2_SH(vec1, vec0, vec3, vec2, vec9, vec10);

        CLIP_SH2_0_255(vec9, vec10);

        PCKEV_ST_SB(vec9, vec10, src);

        src += stride;



        vec4 += vec5;

    }

}
