void idct_put_altivec(uint8_t* dest, int stride, vector_s16_t* block)

{

POWERPC_TBL_DECLARE(altivec_idct_put_num, 1);

#ifdef ALTIVEC_USE_REFERENCE_C_CODE

POWERPC_TBL_START_COUNT(altivec_idct_put_num, 1);

    void simple_idct_put(uint8_t *dest, int line_size, int16_t *block);

    simple_idct_put(dest, stride, (int16_t*)block);

POWERPC_TBL_STOP_COUNT(altivec_idct_put_num, 1);

#else /* ALTIVEC_USE_REFERENCE_C_CODE */

    vector_u8_t tmp;



POWERPC_TBL_START_COUNT(altivec_idct_put_num, 1);



    IDCT



#define COPY(dest,src)						\

    tmp = vec_packsu (src, src);				\

    vec_ste ((vector_u32_t)tmp, 0, (unsigned int *)dest);	\

    vec_ste ((vector_u32_t)tmp, 4, (unsigned int *)dest);



    COPY (dest, vx0)	dest += stride;

    COPY (dest, vx1)	dest += stride;

    COPY (dest, vx2)	dest += stride;

    COPY (dest, vx3)	dest += stride;

    COPY (dest, vx4)	dest += stride;

    COPY (dest, vx5)	dest += stride;

    COPY (dest, vx6)	dest += stride;

    COPY (dest, vx7)



POWERPC_TBL_STOP_COUNT(altivec_idct_put_num, 1);

#endif /* ALTIVEC_USE_REFERENCE_C_CODE */

}
