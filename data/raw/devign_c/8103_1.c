static void diff_pixels_altivec(int16_t *restrict block, const uint8_t *s1,

                                const uint8_t *s2, int stride)

{

    int i;

    vec_u8 perm1 = vec_lvsl(0, s1);

    vec_u8 perm2 = vec_lvsl(0, s2);

    const vec_u8 zero = (const vec_u8)vec_splat_u8(0);

    vec_s16 shorts1, shorts2;



    for (i = 0; i < 4; i++) {

        /* Read potentially unaligned pixels.

         * We're reading 16 pixels, and actually only want 8,

         * but we simply ignore the extras. */

        vec_u8 pixl  = vec_ld(0,  s1);

        vec_u8 pixr  = vec_ld(15, s1);

        vec_u8 bytes = vec_perm(pixl, pixr, perm1);



        // Convert the bytes into shorts.

        shorts1 = (vec_s16)vec_mergeh(zero, bytes);



        // Do the same for the second block of pixels.

        pixl  = vec_ld(0,  s2);

        pixr  = vec_ld(15, s2);

        bytes = vec_perm(pixl, pixr, perm2);



        // Convert the bytes into shorts.

        shorts2 = (vec_s16)vec_mergeh(zero, bytes);



        // Do the subtraction.

        shorts1 = vec_sub(shorts1, shorts2);



        // Save the data to the block, we assume the block is 16-byte aligned.

        vec_st(shorts1, 0, (vec_s16 *)block);



        s1    += stride;

        s2    += stride;

        block += 8;



        /* The code below is a copy of the code above...

         * This is a manual unroll. */



        /* Read potentially unaligned pixels.

         * We're reading 16 pixels, and actually only want 8,

         * but we simply ignore the extras. */

        pixl  = vec_ld(0,  s1);

        pixr  = vec_ld(15, s1);

        bytes = vec_perm(pixl, pixr, perm1);



        // Convert the bytes into shorts.

        shorts1 = (vec_s16)vec_mergeh(zero, bytes);



        // Do the same for the second block of pixels.

        pixl  = vec_ld(0,  s2);

        pixr  = vec_ld(15, s2);

        bytes = vec_perm(pixl, pixr, perm2);



        // Convert the bytes into shorts.

        shorts2 = (vec_s16)vec_mergeh(zero, bytes);



        // Do the subtraction.

        shorts1 = vec_sub(shorts1, shorts2);



        // Save the data to the block, we assume the block is 16-byte aligned.

        vec_st(shorts1, 0, (vec_s16 *)block);



        s1    += stride;

        s2    += stride;

        block += 8;

    }

}
