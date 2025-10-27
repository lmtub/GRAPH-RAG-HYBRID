void dsputil_init_ppc(DSPContext* c, AVCodecContext *avctx)

{

    // Common optimizations whether Altivec is available or not



  switch (check_dcbzl_effect()) {

  case 32:

    c->clear_blocks = clear_blocks_dcbz32_ppc;

    break;

  case 128:

    c->clear_blocks = clear_blocks_dcbz128_ppc;

    break;

  default:

    break;

  }

  

#if HAVE_ALTIVEC

    if (has_altivec()) {

        mm_flags |= MM_ALTIVEC;

        

        // Altivec specific optimisations

        c->pix_abs16x16_x2 = pix_abs16x16_x2_altivec;

        c->pix_abs16x16_y2 = pix_abs16x16_y2_altivec;

        c->pix_abs16x16_xy2 = pix_abs16x16_xy2_altivec;

        c->pix_abs16x16 = pix_abs16x16_altivec;

        c->pix_abs8x8 = pix_abs8x8_altivec;

        c->sad[0]= sad16x16_altivec;

        c->sad[1]= sad8x8_altivec;

        c->pix_norm1 = pix_norm1_altivec;

        c->sse[1]= sse8_altivec;

        c->sse[0]= sse16_altivec;

        c->pix_sum = pix_sum_altivec;

        c->diff_pixels = diff_pixels_altivec;

        c->get_pixels = get_pixels_altivec;

// next one disabled as it's untested.

#if 0

        c->add_bytes= add_bytes_altivec;

#endif /* 0 */

        c->put_pixels_tab[0][0] = put_pixels16_altivec;

        c->avg_pixels_tab[0][0] = avg_pixels16_altivec;

// next one disabled as it's untested.

#if 0

        c->avg_pixels_tab[1][0] = avg_pixels8_altivec;

#endif /* 0 */

        c->put_pixels_tab[1][3] = put_pixels8_xy2_altivec;

        c->put_no_rnd_pixels_tab[1][3] = put_no_rnd_pixels8_xy2_altivec;

        c->put_pixels_tab[0][3] = put_pixels16_xy2_altivec;

        c->put_no_rnd_pixels_tab[0][3] = put_no_rnd_pixels16_xy2_altivec;

        

	c->gmc1 = gmc1_altivec;



        if ((avctx->idct_algo == FF_IDCT_AUTO) ||

                (avctx->idct_algo == FF_IDCT_ALTIVEC))

        {

            c->idct_put = idct_put_altivec;

            c->idct_add = idct_add_altivec;

#ifndef ALTIVEC_USE_REFERENCE_C_CODE

            c->idct_permutation_type = FF_TRANSPOSE_IDCT_PERM;

#else /* ALTIVEC_USE_REFERENCE_C_CODE */

            c->idct_permutation_type = FF_NO_IDCT_PERM;

#endif /* ALTIVEC_USE_REFERENCE_C_CODE */

        }

        

#ifdef POWERPC_TBL_PERFORMANCE_REPORT

        {

          int i;

          for (i = 0 ; i < powerpc_perf_total ; i++)

          {

            perfdata[i][powerpc_data_min] = 0xFFFFFFFFFFFFFFFF;

            perfdata[i][powerpc_data_max] = 0x0000000000000000;

            perfdata[i][powerpc_data_sum] = 0x0000000000000000;

            perfdata[i][powerpc_data_num] = 0x0000000000000000;

#ifdef POWERPC_PERF_USE_PMC

            perfdata_pmc2[i][powerpc_data_min] = 0xFFFFFFFFFFFFFFFF;

            perfdata_pmc2[i][powerpc_data_max] = 0x0000000000000000;

            perfdata_pmc2[i][powerpc_data_sum] = 0x0000000000000000;

            perfdata_pmc2[i][powerpc_data_num] = 0x0000000000000000;

            perfdata_pmc3[i][powerpc_data_min] = 0xFFFFFFFFFFFFFFFF;

            perfdata_pmc3[i][powerpc_data_max] = 0x0000000000000000;

            perfdata_pmc3[i][powerpc_data_sum] = 0x0000000000000000;

            perfdata_pmc3[i][powerpc_data_num] = 0x0000000000000000;

#endif /* POWERPC_PERF_USE_PMC */

          }

        }

#endif /* POWERPC_TBL_PERFORMANCE_REPORT */

    } else

#endif /* HAVE_ALTIVEC */

    {

        // Non-AltiVec PPC optimisations



        // ... pending ...

    }

}
