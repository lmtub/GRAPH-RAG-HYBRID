void dsputil_init(DSPContext* c, AVCodecContext *avctx)

{

    int i;



    ff_check_alignment();



#if CONFIG_ENCODERS

    if(avctx->dct_algo==FF_DCT_FASTINT) {

        c->fdct = fdct_ifast;

        c->fdct248 = fdct_ifast248;

    }

    else if(avctx->dct_algo==FF_DCT_FAAN) {

        c->fdct = ff_faandct;

        c->fdct248 = ff_faandct248;

    }

    else {

        c->fdct = ff_jpeg_fdct_islow; //slow/accurate/default

        c->fdct248 = ff_fdct248_islow;

    }

#endif //CONFIG_ENCODERS



    if(avctx->lowres==1){

        if(avctx->idct_algo==FF_IDCT_INT || avctx->idct_algo==FF_IDCT_AUTO || !CONFIG_H264_DECODER){

            c->idct_put= ff_jref_idct4_put;

            c->idct_add= ff_jref_idct4_add;

        }else{

            c->idct_put= ff_h264_lowres_idct_put_c;

            c->idct_add= ff_h264_lowres_idct_add_c;

        }

        c->idct    = j_rev_dct4;

        c->idct_permutation_type= FF_NO_IDCT_PERM;

    }else if(avctx->lowres==2){

        c->idct_put= ff_jref_idct2_put;

        c->idct_add= ff_jref_idct2_add;

        c->idct    = j_rev_dct2;

        c->idct_permutation_type= FF_NO_IDCT_PERM;

    }else if(avctx->lowres==3){

        c->idct_put= ff_jref_idct1_put;

        c->idct_add= ff_jref_idct1_add;

        c->idct    = j_rev_dct1;

        c->idct_permutation_type= FF_NO_IDCT_PERM;

    }else{

        if(avctx->idct_algo==FF_IDCT_INT){

            c->idct_put= ff_jref_idct_put;

            c->idct_add= ff_jref_idct_add;

            c->idct    = j_rev_dct;

            c->idct_permutation_type= FF_LIBMPEG2_IDCT_PERM;

        }else if((CONFIG_VP3_DECODER || CONFIG_VP5_DECODER || CONFIG_VP6_DECODER ) &&

                avctx->idct_algo==FF_IDCT_VP3){

            c->idct_put= ff_vp3_idct_put_c;

            c->idct_add= ff_vp3_idct_add_c;

            c->idct    = ff_vp3_idct_c;

            c->idct_permutation_type= FF_NO_IDCT_PERM;

        }else if(avctx->idct_algo==FF_IDCT_WMV2){

            c->idct_put= ff_wmv2_idct_put_c;

            c->idct_add= ff_wmv2_idct_add_c;

            c->idct    = ff_wmv2_idct_c;

            c->idct_permutation_type= FF_NO_IDCT_PERM;

        }else if(avctx->idct_algo==FF_IDCT_FAAN){

            c->idct_put= ff_faanidct_put;

            c->idct_add= ff_faanidct_add;

            c->idct    = ff_faanidct;

            c->idct_permutation_type= FF_NO_IDCT_PERM;

        }else if(CONFIG_EATGQ_DECODER && avctx->idct_algo==FF_IDCT_EA) {

            c->idct_put= ff_ea_idct_put_c;

            c->idct_permutation_type= FF_NO_IDCT_PERM;

        }else{ //accurate/default

            c->idct_put= ff_simple_idct_put;

            c->idct_add= ff_simple_idct_add;

            c->idct    = ff_simple_idct;

            c->idct_permutation_type= FF_NO_IDCT_PERM;

        }

    }



    if (CONFIG_H264_DECODER) {

        c->h264_idct_add= ff_h264_idct_add_c;

        c->h264_idct8_add= ff_h264_idct8_add_c;

        c->h264_idct_dc_add= ff_h264_idct_dc_add_c;

        c->h264_idct8_dc_add= ff_h264_idct8_dc_add_c;

        c->h264_idct_add16     = ff_h264_idct_add16_c;

        c->h264_idct8_add4     = ff_h264_idct8_add4_c;

        c->h264_idct_add8      = ff_h264_idct_add8_c;

        c->h264_idct_add16intra= ff_h264_idct_add16intra_c;

    }



    c->get_pixels = get_pixels_c;

    c->diff_pixels = diff_pixels_c;

    c->put_pixels_clamped = put_pixels_clamped_c;

    c->put_signed_pixels_clamped = put_signed_pixels_clamped_c;

    c->add_pixels_clamped = add_pixels_clamped_c;

    c->add_pixels8 = add_pixels8_c;

    c->add_pixels4 = add_pixels4_c;

    c->sum_abs_dctelem = sum_abs_dctelem_c;

    c->gmc1 = gmc1_c;

    c->gmc = ff_gmc_c;

    c->clear_block = clear_block_c;

    c->clear_blocks = clear_blocks_c;

    c->pix_sum = pix_sum_c;

    c->pix_norm1 = pix_norm1_c;



    /* TODO [0] 16  [1] 8 */

    c->pix_abs[0][0] = pix_abs16_c;

    c->pix_abs[0][1] = pix_abs16_x2_c;

    c->pix_abs[0][2] = pix_abs16_y2_c;

    c->pix_abs[0][3] = pix_abs16_xy2_c;

    c->pix_abs[1][0] = pix_abs8_c;

    c->pix_abs[1][1] = pix_abs8_x2_c;

    c->pix_abs[1][2] = pix_abs8_y2_c;

    c->pix_abs[1][3] = pix_abs8_xy2_c;



#define dspfunc(PFX, IDX, NUM) \

    c->PFX ## _pixels_tab[IDX][0] = PFX ## _pixels ## NUM ## _c;     \

    c->PFX ## _pixels_tab[IDX][1] = PFX ## _pixels ## NUM ## _x2_c;  \

    c->PFX ## _pixels_tab[IDX][2] = PFX ## _pixels ## NUM ## _y2_c;  \

    c->PFX ## _pixels_tab[IDX][3] = PFX ## _pixels ## NUM ## _xy2_c



    dspfunc(put, 0, 16);

    dspfunc(put_no_rnd, 0, 16);

    dspfunc(put, 1, 8);

    dspfunc(put_no_rnd, 1, 8);

    dspfunc(put, 2, 4);

    dspfunc(put, 3, 2);



    dspfunc(avg, 0, 16);

    dspfunc(avg_no_rnd, 0, 16);

    dspfunc(avg, 1, 8);

    dspfunc(avg_no_rnd, 1, 8);

    dspfunc(avg, 2, 4);

    dspfunc(avg, 3, 2);

#undef dspfunc



    c->put_no_rnd_pixels_l2[0]= put_no_rnd_pixels16_l2_c;

    c->put_no_rnd_pixels_l2[1]= put_no_rnd_pixels8_l2_c;



    c->put_tpel_pixels_tab[ 0] = put_tpel_pixels_mc00_c;

    c->put_tpel_pixels_tab[ 1] = put_tpel_pixels_mc10_c;

    c->put_tpel_pixels_tab[ 2] = put_tpel_pixels_mc20_c;

    c->put_tpel_pixels_tab[ 4] = put_tpel_pixels_mc01_c;

    c->put_tpel_pixels_tab[ 5] = put_tpel_pixels_mc11_c;

    c->put_tpel_pixels_tab[ 6] = put_tpel_pixels_mc21_c;

    c->put_tpel_pixels_tab[ 8] = put_tpel_pixels_mc02_c;

    c->put_tpel_pixels_tab[ 9] = put_tpel_pixels_mc12_c;

    c->put_tpel_pixels_tab[10] = put_tpel_pixels_mc22_c;



    c->avg_tpel_pixels_tab[ 0] = avg_tpel_pixels_mc00_c;

    c->avg_tpel_pixels_tab[ 1] = avg_tpel_pixels_mc10_c;

    c->avg_tpel_pixels_tab[ 2] = avg_tpel_pixels_mc20_c;

    c->avg_tpel_pixels_tab[ 4] = avg_tpel_pixels_mc01_c;

    c->avg_tpel_pixels_tab[ 5] = avg_tpel_pixels_mc11_c;

    c->avg_tpel_pixels_tab[ 6] = avg_tpel_pixels_mc21_c;

    c->avg_tpel_pixels_tab[ 8] = avg_tpel_pixels_mc02_c;

    c->avg_tpel_pixels_tab[ 9] = avg_tpel_pixels_mc12_c;

    c->avg_tpel_pixels_tab[10] = avg_tpel_pixels_mc22_c;



#define dspfunc(PFX, IDX, NUM) \

    c->PFX ## _pixels_tab[IDX][ 0] = PFX ## NUM ## _mc00_c; \

    c->PFX ## _pixels_tab[IDX][ 1] = PFX ## NUM ## _mc10_c; \

    c->PFX ## _pixels_tab[IDX][ 2] = PFX ## NUM ## _mc20_c; \

    c->PFX ## _pixels_tab[IDX][ 3] = PFX ## NUM ## _mc30_c; \

    c->PFX ## _pixels_tab[IDX][ 4] = PFX ## NUM ## _mc01_c; \

    c->PFX ## _pixels_tab[IDX][ 5] = PFX ## NUM ## _mc11_c; \

    c->PFX ## _pixels_tab[IDX][ 6] = PFX ## NUM ## _mc21_c; \

    c->PFX ## _pixels_tab[IDX][ 7] = PFX ## NUM ## _mc31_c; \

    c->PFX ## _pixels_tab[IDX][ 8] = PFX ## NUM ## _mc02_c; \

    c->PFX ## _pixels_tab[IDX][ 9] = PFX ## NUM ## _mc12_c; \

    c->PFX ## _pixels_tab[IDX][10] = PFX ## NUM ## _mc22_c; \

    c->PFX ## _pixels_tab[IDX][11] = PFX ## NUM ## _mc32_c; \

    c->PFX ## _pixels_tab[IDX][12] = PFX ## NUM ## _mc03_c; \

    c->PFX ## _pixels_tab[IDX][13] = PFX ## NUM ## _mc13_c; \

    c->PFX ## _pixels_tab[IDX][14] = PFX ## NUM ## _mc23_c; \

    c->PFX ## _pixels_tab[IDX][15] = PFX ## NUM ## _mc33_c



    dspfunc(put_qpel, 0, 16);

    dspfunc(put_no_rnd_qpel, 0, 16);



    dspfunc(avg_qpel, 0, 16);

    /* dspfunc(avg_no_rnd_qpel, 0, 16); */



    dspfunc(put_qpel, 1, 8);

    dspfunc(put_no_rnd_qpel, 1, 8);



    dspfunc(avg_qpel, 1, 8);

    /* dspfunc(avg_no_rnd_qpel, 1, 8); */



    dspfunc(put_h264_qpel, 0, 16);

    dspfunc(put_h264_qpel, 1, 8);

    dspfunc(put_h264_qpel, 2, 4);

    dspfunc(put_h264_qpel, 3, 2);

    dspfunc(avg_h264_qpel, 0, 16);

    dspfunc(avg_h264_qpel, 1, 8);

    dspfunc(avg_h264_qpel, 2, 4);



#undef dspfunc

    c->put_h264_chroma_pixels_tab[0]= put_h264_chroma_mc8_c;

    c->put_h264_chroma_pixels_tab[1]= put_h264_chroma_mc4_c;

    c->put_h264_chroma_pixels_tab[2]= put_h264_chroma_mc2_c;

    c->avg_h264_chroma_pixels_tab[0]= avg_h264_chroma_mc8_c;

    c->avg_h264_chroma_pixels_tab[1]= avg_h264_chroma_mc4_c;

    c->avg_h264_chroma_pixels_tab[2]= avg_h264_chroma_mc2_c;

    c->put_no_rnd_vc1_chroma_pixels_tab[0]= put_no_rnd_vc1_chroma_mc8_c;

    c->avg_no_rnd_vc1_chroma_pixels_tab[0]= avg_no_rnd_vc1_chroma_mc8_c;



    c->weight_h264_pixels_tab[0]= weight_h264_pixels16x16_c;

    c->weight_h264_pixels_tab[1]= weight_h264_pixels16x8_c;

    c->weight_h264_pixels_tab[2]= weight_h264_pixels8x16_c;

    c->weight_h264_pixels_tab[3]= weight_h264_pixels8x8_c;

    c->weight_h264_pixels_tab[4]= weight_h264_pixels8x4_c;

    c->weight_h264_pixels_tab[5]= weight_h264_pixels4x8_c;

    c->weight_h264_pixels_tab[6]= weight_h264_pixels4x4_c;

    c->weight_h264_pixels_tab[7]= weight_h264_pixels4x2_c;

    c->weight_h264_pixels_tab[8]= weight_h264_pixels2x4_c;

    c->weight_h264_pixels_tab[9]= weight_h264_pixels2x2_c;

    c->biweight_h264_pixels_tab[0]= biweight_h264_pixels16x16_c;

    c->biweight_h264_pixels_tab[1]= biweight_h264_pixels16x8_c;

    c->biweight_h264_pixels_tab[2]= biweight_h264_pixels8x16_c;

    c->biweight_h264_pixels_tab[3]= biweight_h264_pixels8x8_c;

    c->biweight_h264_pixels_tab[4]= biweight_h264_pixels8x4_c;

    c->biweight_h264_pixels_tab[5]= biweight_h264_pixels4x8_c;

    c->biweight_h264_pixels_tab[6]= biweight_h264_pixels4x4_c;

    c->biweight_h264_pixels_tab[7]= biweight_h264_pixels4x2_c;

    c->biweight_h264_pixels_tab[8]= biweight_h264_pixels2x4_c;

    c->biweight_h264_pixels_tab[9]= biweight_h264_pixels2x2_c;



    c->draw_edges = draw_edges_c;



#if CONFIG_CAVS_DECODER

    ff_cavsdsp_init(c,avctx);

#endif



#if CONFIG_MLP_DECODER || CONFIG_TRUEHD_DECODER

    ff_mlp_init(c, avctx);

#endif

#if CONFIG_VC1_DECODER || CONFIG_WMV3_DECODER

    ff_vc1dsp_init(c,avctx);

#endif

#if CONFIG_WMV2_DECODER || CONFIG_VC1_DECODER || CONFIG_WMV3_DECODER

    ff_intrax8dsp_init(c,avctx);

#endif

#if CONFIG_RV30_DECODER

    ff_rv30dsp_init(c,avctx);

#endif

#if CONFIG_RV40_DECODER

    ff_rv40dsp_init(c,avctx);

    c->put_rv40_qpel_pixels_tab[0][15] = put_rv40_qpel16_mc33_c;

    c->avg_rv40_qpel_pixels_tab[0][15] = avg_rv40_qpel16_mc33_c;

    c->put_rv40_qpel_pixels_tab[1][15] = put_rv40_qpel8_mc33_c;

    c->avg_rv40_qpel_pixels_tab[1][15] = avg_rv40_qpel8_mc33_c;

#endif



    c->put_mspel_pixels_tab[0]= put_mspel8_mc00_c;

    c->put_mspel_pixels_tab[1]= put_mspel8_mc10_c;

    c->put_mspel_pixels_tab[2]= put_mspel8_mc20_c;

    c->put_mspel_pixels_tab[3]= put_mspel8_mc30_c;

    c->put_mspel_pixels_tab[4]= put_mspel8_mc02_c;

    c->put_mspel_pixels_tab[5]= put_mspel8_mc12_c;

    c->put_mspel_pixels_tab[6]= put_mspel8_mc22_c;

    c->put_mspel_pixels_tab[7]= put_mspel8_mc32_c;



#define SET_CMP_FUNC(name) \

    c->name[0]= name ## 16_c;\

    c->name[1]= name ## 8x8_c;



    SET_CMP_FUNC(hadamard8_diff)

    c->hadamard8_diff[4]= hadamard8_intra16_c;

    c->hadamard8_diff[5]= hadamard8_intra8x8_c;

    SET_CMP_FUNC(dct_sad)

    SET_CMP_FUNC(dct_max)

#if CONFIG_GPL

    SET_CMP_FUNC(dct264_sad)

#endif

    c->sad[0]= pix_abs16_c;

    c->sad[1]= pix_abs8_c;

    c->sse[0]= sse16_c;

    c->sse[1]= sse8_c;

    c->sse[2]= sse4_c;

    SET_CMP_FUNC(quant_psnr)

    SET_CMP_FUNC(rd)

    SET_CMP_FUNC(bit)

    c->vsad[0]= vsad16_c;

    c->vsad[4]= vsad_intra16_c;

    c->vsad[5]= vsad_intra8_c;

    c->vsse[0]= vsse16_c;

    c->vsse[4]= vsse_intra16_c;

    c->vsse[5]= vsse_intra8_c;

    c->nsse[0]= nsse16_c;

    c->nsse[1]= nsse8_c;

#if CONFIG_SNOW_ENCODER

    c->w53[0]= w53_16_c;

    c->w53[1]= w53_8_c;

    c->w97[0]= w97_16_c;

    c->w97[1]= w97_8_c;

#endif



    c->ssd_int8_vs_int16 = ssd_int8_vs_int16_c;



    c->add_bytes= add_bytes_c;

    c->add_bytes_l2= add_bytes_l2_c;

    c->diff_bytes= diff_bytes_c;

    c->add_hfyu_median_prediction= add_hfyu_median_prediction_c;

    c->sub_hfyu_median_prediction= sub_hfyu_median_prediction_c;

    c->bswap_buf= bswap_buf;

#if CONFIG_PNG_DECODER

    c->add_png_paeth_prediction= ff_add_png_paeth_prediction;

#endif



    c->h264_v_loop_filter_luma= h264_v_loop_filter_luma_c;

    c->h264_h_loop_filter_luma= h264_h_loop_filter_luma_c;

    c->h264_v_loop_filter_luma_intra= h264_v_loop_filter_luma_intra_c;

    c->h264_h_loop_filter_luma_intra= h264_h_loop_filter_luma_intra_c;

    c->h264_v_loop_filter_chroma= h264_v_loop_filter_chroma_c;

    c->h264_h_loop_filter_chroma= h264_h_loop_filter_chroma_c;

    c->h264_v_loop_filter_chroma_intra= h264_v_loop_filter_chroma_intra_c;

    c->h264_h_loop_filter_chroma_intra= h264_h_loop_filter_chroma_intra_c;

    c->h264_loop_filter_strength= NULL;



    if (CONFIG_ANY_H263) {

        c->h263_h_loop_filter= h263_h_loop_filter_c;

        c->h263_v_loop_filter= h263_v_loop_filter_c;

    }



    if (CONFIG_VP3_DECODER) {

        c->vp3_h_loop_filter= ff_vp3_h_loop_filter_c;

        c->vp3_v_loop_filter= ff_vp3_v_loop_filter_c;

    }

    if (CONFIG_VP6_DECODER) {

        c->vp6_filter_diag4= ff_vp6_filter_diag4_c;

    }



    c->h261_loop_filter= h261_loop_filter_c;



    c->try_8x8basis= try_8x8basis_c;

    c->add_8x8basis= add_8x8basis_c;



#if CONFIG_SNOW_DECODER

    c->vertical_compose97i = ff_snow_vertical_compose97i;

    c->horizontal_compose97i = ff_snow_horizontal_compose97i;

    c->inner_add_yblock = ff_snow_inner_add_yblock;

#endif



#if CONFIG_VORBIS_DECODER

    c->vorbis_inverse_coupling = vorbis_inverse_coupling;

#endif

#if CONFIG_AC3_DECODER

    c->ac3_downmix = ff_ac3_downmix_c;

#endif

#if CONFIG_FLAC_ENCODER

    c->flac_compute_autocorr = ff_flac_compute_autocorr;

#endif

    c->vector_fmul = vector_fmul_c;

    c->vector_fmul_reverse = vector_fmul_reverse_c;

    c->vector_fmul_add_add = ff_vector_fmul_add_add_c;

    c->vector_fmul_window = ff_vector_fmul_window_c;

    c->int32_to_float_fmul_scalar = int32_to_float_fmul_scalar_c;

    c->float_to_int16 = ff_float_to_int16_c;

    c->float_to_int16_interleave = ff_float_to_int16_interleave_c;

    c->add_int16 = add_int16_c;

    c->sub_int16 = sub_int16_c;

    c->scalarproduct_int16 = scalarproduct_int16_c;



    c->shrink[0]= ff_img_copy_plane;

    c->shrink[1]= ff_shrink22;

    c->shrink[2]= ff_shrink44;

    c->shrink[3]= ff_shrink88;



    c->prefetch= just_return;



    memset(c->put_2tap_qpel_pixels_tab, 0, sizeof(c->put_2tap_qpel_pixels_tab));

    memset(c->avg_2tap_qpel_pixels_tab, 0, sizeof(c->avg_2tap_qpel_pixels_tab));



    if (HAVE_MMX)        dsputil_init_mmx   (c, avctx);

    if (ARCH_ARM)        dsputil_init_arm   (c, avctx);

    if (CONFIG_MLIB)     dsputil_init_mlib  (c, avctx);

    if (HAVE_VIS)        dsputil_init_vis   (c, avctx);

    if (ARCH_ALPHA)      dsputil_init_alpha (c, avctx);

    if (ARCH_PPC)        dsputil_init_ppc   (c, avctx);

    if (HAVE_MMI)        dsputil_init_mmi   (c, avctx);

    if (ARCH_SH4)        dsputil_init_sh4   (c, avctx);

    if (ARCH_BFIN)       dsputil_init_bfin  (c, avctx);



    for(i=0; i<64; i++){

        if(!c->put_2tap_qpel_pixels_tab[0][i])

            c->put_2tap_qpel_pixels_tab[0][i]= c->put_h264_qpel_pixels_tab[0][i];

        if(!c->avg_2tap_qpel_pixels_tab[0][i])

            c->avg_2tap_qpel_pixels_tab[0][i]= c->avg_h264_qpel_pixels_tab[0][i];

    }



    switch(c->idct_permutation_type){

    case FF_NO_IDCT_PERM:

        for(i=0; i<64; i++)

            c->idct_permutation[i]= i;

        break;

    case FF_LIBMPEG2_IDCT_PERM:

        for(i=0; i<64; i++)

            c->idct_permutation[i]= (i & 0x38) | ((i & 6) >> 1) | ((i & 1) << 2);

        break;

    case FF_SIMPLE_IDCT_PERM:

        for(i=0; i<64; i++)

            c->idct_permutation[i]= simple_mmx_permutation[i];

        break;

    case FF_TRANSPOSE_IDCT_PERM:

        for(i=0; i<64; i++)

            c->idct_permutation[i]= ((i&7)<<3) | (i>>3);

        break;

    case FF_PARTTRANS_IDCT_PERM:

        for(i=0; i<64; i++)

            c->idct_permutation[i]= (i&0x24) | ((i&3)<<3) | ((i>>3)&3);

        break;

    case FF_SSE2_IDCT_PERM:

        for(i=0; i<64; i++)

            c->idct_permutation[i]= (i&0x38) | idct_sse2_row_perm[i&7];

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "Internal error, IDCT permutation not set\n");

    }

}
