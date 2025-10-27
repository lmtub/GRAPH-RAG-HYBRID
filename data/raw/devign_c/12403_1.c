void put_vp8_epel_h_altivec_core(uint8_t *dst, int dst_stride,

                                 uint8_t *src, int src_stride,

                                 int h, int mx, int w, int is6tap)

{

    LOAD_H_SUBPEL_FILTER(mx-1);

    vec_u8 align_vec0, align_vec8, permh0, permh8, filt;

    vec_u8 perm_6tap0, perm_6tap8, perml0, perml8;

    vec_u8 a, b, pixh, pixl, outer;

    vec_s16 f16h, f16l;

    vec_s32 filth, filtl;



    vec_u8 perm_inner = { 1,2,3,4, 2,3,4,5, 3,4,5,6, 4,5,6,7 };

    vec_u8 perm_outer = { 4,9, 0,5, 5,10, 1,6, 6,11, 2,7, 7,12, 3,8 };

    vec_s32 c64 = vec_sl(vec_splat_s32(1), vec_splat_u32(6));

    vec_u16 c7  = vec_splat_u16(7);



    align_vec0 = vec_lvsl( -2, src);

    align_vec8 = vec_lvsl(8-2, src);



    permh0     = vec_perm(align_vec0, align_vec0, perm_inner);

    permh8     = vec_perm(align_vec8, align_vec8, perm_inner);

    perm_inner = vec_add(perm_inner, vec_splat_u8(4));

    perml0     = vec_perm(align_vec0, align_vec0, perm_inner);

    perml8     = vec_perm(align_vec8, align_vec8, perm_inner);

    perm_6tap0 = vec_perm(align_vec0, align_vec0, perm_outer);

    perm_6tap8 = vec_perm(align_vec8, align_vec8, perm_outer);



    while (h --> 0) {

        FILTER_H(f16h, 0);



        if (w == 16) {

            FILTER_H(f16l, 8);

            filt = vec_packsu(f16h, f16l);

            vec_st(filt, 0, dst);

        } else {

            filt = vec_packsu(f16h, f16h);

            vec_ste((vec_u32)filt, 0, (uint32_t*)dst);

            if (w == 8)

                vec_ste((vec_u32)filt, 4, (uint32_t*)dst);

        }

        src += src_stride;

        dst += dst_stride;

    }

}
