void dsputil_init_ppc(void)

{

    // Common optimisations whether Altivec or not



    // ... pending ...



#if HAVE_ALTIVEC

    if (has_altivec()) {

        // Altivec specific optimisations

        pix_abs16x16 = pix_abs16x16_altivec;

        pix_abs8x8 = pix_abs8x8_altivec;

        pix_sum = pix_sum_altivec;

        diff_pixels = diff_pixels_altivec;

        get_pixels = get_pixels_altivec;

    } else

#endif

    {

        // Non-AltiVec PPC optimisations



        // ... pending ...

    }

}
