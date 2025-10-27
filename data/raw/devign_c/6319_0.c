int DCT_common_init(MpegEncContext *s)

{

    int i;



    ff_put_pixels_clamped = s->dsp.put_pixels_clamped;

    ff_add_pixels_clamped = s->dsp.add_pixels_clamped;



    s->dct_unquantize_h263 = dct_unquantize_h263_c;

    s->dct_unquantize_mpeg1 = dct_unquantize_mpeg1_c;

    s->dct_unquantize_mpeg2 = dct_unquantize_mpeg2_c;

    s->dct_quantize= dct_quantize_c;



    if(s->avctx->dct_algo==FF_DCT_FASTINT)

        s->fdct = fdct_ifast;

    else

        s->fdct = ff_jpeg_fdct_islow; //slow/accurate/default



    if(s->avctx->idct_algo==FF_IDCT_INT){

        s->idct_put= ff_jref_idct_put;

        s->idct_add= ff_jref_idct_add;

        s->idct_permutation_type= FF_LIBMPEG2_IDCT_PERM;

    }else{ //accurate/default

        s->idct_put= simple_idct_put;

        s->idct_add= simple_idct_add;

        s->idct_permutation_type= FF_NO_IDCT_PERM;

    }

        

#ifdef HAVE_MMX

    MPV_common_init_mmx(s);

#endif

#ifdef ARCH_ALPHA

    MPV_common_init_axp(s);

#endif

#ifdef HAVE_MLIB

    MPV_common_init_mlib(s);

#endif

#ifdef HAVE_MMI

    MPV_common_init_mmi(s);

#endif

#ifdef ARCH_ARMV4L

    MPV_common_init_armv4l();

#endif

#ifdef ARCH_POWERPC

    MPV_common_init_ppc(s);

#endif



    switch(s->idct_permutation_type){

    case FF_NO_IDCT_PERM:

        for(i=0; i<64; i++)

            s->idct_permutation[i]= i;

        break;

    case FF_LIBMPEG2_IDCT_PERM:

        for(i=0; i<64; i++)

            s->idct_permutation[i]= (i & 0x38) | ((i & 6) >> 1) | ((i & 1) << 2);

        break;

    case FF_SIMPLE_IDCT_PERM:

        for(i=0; i<64; i++)

            s->idct_permutation[i]= simple_mmx_permutation[i];

        break;

    case FF_TRANSPOSE_IDCT_PERM:

        for(i=0; i<64; i++)

            s->idct_permutation[i]= ((i&7)<<3) | (i>>3);

        break;

    default:

        fprintf(stderr, "Internal error, IDCT permutation not set\n");

        return -1;

    }





    /* load & permutate scantables

       note: only wmv uses differnt ones 

    */

    ff_init_scantable(s, &s->inter_scantable  , ff_zigzag_direct);

    ff_init_scantable(s, &s->intra_scantable  , ff_zigzag_direct);

    ff_init_scantable(s, &s->intra_h_scantable, ff_alternate_horizontal_scan);

    ff_init_scantable(s, &s->intra_v_scantable, ff_alternate_vertical_scan);



    return 0;

}
