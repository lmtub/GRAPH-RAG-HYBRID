static av_cold int dct_init(MpegEncContext *s)

{

    ff_blockdsp_init(&s->bdsp, s->avctx);

    ff_hpeldsp_init(&s->hdsp, s->avctx->flags);

    ff_me_cmp_init(&s->mecc, s->avctx);

    ff_mpegvideodsp_init(&s->mdsp);

    ff_videodsp_init(&s->vdsp, s->avctx->bits_per_raw_sample);



    s->dct_unquantize_h263_intra = dct_unquantize_h263_intra_c;

    s->dct_unquantize_h263_inter = dct_unquantize_h263_inter_c;

    s->dct_unquantize_mpeg1_intra = dct_unquantize_mpeg1_intra_c;

    s->dct_unquantize_mpeg1_inter = dct_unquantize_mpeg1_inter_c;

    s->dct_unquantize_mpeg2_intra = dct_unquantize_mpeg2_intra_c;

    if (s->flags & CODEC_FLAG_BITEXACT)

        s->dct_unquantize_mpeg2_intra = dct_unquantize_mpeg2_intra_bitexact;

    s->dct_unquantize_mpeg2_inter = dct_unquantize_mpeg2_inter_c;



    if (HAVE_INTRINSICS_NEON)

        ff_mpv_common_init_neon(s);



    if (ARCH_ARM)

        ff_mpv_common_init_arm(s);

    if (ARCH_PPC)

        ff_mpv_common_init_ppc(s);

    if (ARCH_X86)

        ff_mpv_common_init_x86(s);



    return 0;

}
