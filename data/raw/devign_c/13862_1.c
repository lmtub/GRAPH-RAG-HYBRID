av_cold void ff_dsputil_init(DSPContext* c, AVCodecContext *avctx)

{

    int i;



    ff_check_alignment();



#if CONFIG_ENCODERS

    if (avctx->bits_per_raw_sample == 10) {

        c->fdct    = ff_jpeg_fdct_islow_10;

        c->fdct248 = ff_fdct248_islow_10;

    } else {

        if(avctx->dct_algo==FF_DCT_FASTINT) {

            c->fdct    = ff_fdct_ifast;

            c->fdct248 = ff_fdct_ifast248;

        }

        else if(avctx->dct_algo==FF_DCT_FAAN) {

            c->fdct    = ff_faandct;

            c->fdct248 = ff_faandct248;

        }

        else {

            c->fdct    = ff_jpeg_fdct_islow_8; //slow/accurate/default

            c->fdct248 = ff_fdct248_islow_8;

        }

    }

#endif //CONFIG_ENCODERS



    if (avctx->bits_per_raw_sample == 10) {

        c->idct_put              = ff_simple_idct_put_10;

        c->idct_add              = ff_simple_idct_add_10;

        c->idct                  = ff_simple_idct_10;

        c->idct_permutation_type = FF_NO_IDCT_PERM;

    } else {

        if(avctx->idct_algo==FF_IDCT_INT){

            c->idct_put= ff_jref_idct_put;

            c->idct_add= ff_jref_idct_add;

            c->idct    = ff_j_rev_dct;

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

            c->idct_put = ff_simple_idct_put_8;

            c->idct_add = ff_simple_idct_add_8;

            c->idct     = ff_simple_idct_8;

            c->idct_permutation_type= FF_NO_IDCT_PERM;

        }

    }



    c->diff_pixels = diff_pixels_c;

    c->put_pixels_clamped = ff_put_pixels_clamped_c;

    c->put_signed_pixels_clamped = ff_put_signed_pixels_clamped_c;

    c->add_pixels_clamped = ff_add_pixels_clamped_c;

    c->sum_abs_dctelem = sum_abs_dctelem_c;

    c->gmc1 = gmc1_c;

    c->gmc = ff_gmc_c;

    c->pix_sum = pix_sum_c;

    c->pix_norm1 = pix_norm1_c;



    c->fill_block_tab[0] = fill_block16_c;

    c->fill_block_tab[1] = fill_block8_c;



    /* TODO [0] 16  [1] 8 */

    c->pix_abs[0][0] = pix_abs16_c;

    c->pix_abs[0][1] = pix_abs16_x2_c;

    c->pix_abs[0][2] = pix_abs16_y2_c;

    c->pix_abs[0][3] = pix_abs16_xy2_c;

    c->pix_abs[1][0] = pix_abs8_c;

    c->pix_abs[1][1] = pix_abs8_x2_c;

    c->pix_abs[1][2] = pix_abs8_y2_c;

    c->pix_abs[1][3] = pix_abs8_xy2_c;



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



#undef dspfunc



#if CONFIG_MLP_DECODER || CONFIG_TRUEHD_DECODER

    ff_mlp_init(c, avctx);

#endif

#if CONFIG_WMV2_DECODER || CONFIG_VC1_DECODER

    ff_intrax8dsp_init(c,avctx);

#endif



    c->put_mspel_pixels_tab[0]= ff_put_pixels8x8_c;

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

#if CONFIG_DWT

    ff_dsputil_init_dwt(c);

#endif



    c->ssd_int8_vs_int16 = ssd_int8_vs_int16_c;



    c->add_bytes= add_bytes_c;

    c->diff_bytes= diff_bytes_c;

    c->add_hfyu_median_prediction= add_hfyu_median_prediction_c;

    c->sub_hfyu_median_prediction= sub_hfyu_median_prediction_c;

    c->add_hfyu_left_prediction  = add_hfyu_left_prediction_c;

    c->add_hfyu_left_prediction_bgr32 = add_hfyu_left_prediction_bgr32_c;

    c->bswap_buf= bswap_buf;

    c->bswap16_buf = bswap16_buf;



    if (CONFIG_H263_DECODER || CONFIG_H263_ENCODER) {

        c->h263_h_loop_filter= h263_h_loop_filter_c;

        c->h263_v_loop_filter= h263_v_loop_filter_c;

    }



    if (CONFIG_VP3_DECODER) {

        c->vp3_h_loop_filter= ff_vp3_h_loop_filter_c;

        c->vp3_v_loop_filter= ff_vp3_v_loop_filter_c;

        c->vp3_idct_dc_add= ff_vp3_idct_dc_add_c;

    }



    c->h261_loop_filter= h261_loop_filter_c;



    c->try_8x8basis= try_8x8basis_c;

    c->add_8x8basis= add_8x8basis_c;



#if CONFIG_VORBIS_DECODER

    c->vorbis_inverse_coupling = ff_vorbis_inverse_coupling;

#endif

#if CONFIG_AC3_DECODER

    c->ac3_downmix = ff_ac3_downmix_c;

#endif

    c->vector_fmul = vector_fmul_c;

    c->vector_fmul_reverse = vector_fmul_reverse_c;

    c->vector_fmul_add = vector_fmul_add_c;

    c->vector_fmul_window = vector_fmul_window_c;

    c->vector_clipf = vector_clipf_c;

    c->scalarproduct_int16 = scalarproduct_int16_c;

    c->scalarproduct_and_madd_int16 = scalarproduct_and_madd_int16_c;

    c->apply_window_int16 = apply_window_int16_c;

    c->vector_clip_int32 = vector_clip_int32_c;

    c->scalarproduct_float = scalarproduct_float_c;

    c->butterflies_float = butterflies_float_c;

    c->butterflies_float_interleave = butterflies_float_interleave_c;

    c->vector_fmul_scalar = vector_fmul_scalar_c;

    c->vector_fmac_scalar = vector_fmac_scalar_c;



    c->shrink[0]= av_image_copy_plane;

    c->shrink[1]= ff_shrink22;

    c->shrink[2]= ff_shrink44;

    c->shrink[3]= ff_shrink88;



    c->prefetch= just_return;



    memset(c->put_2tap_qpel_pixels_tab, 0, sizeof(c->put_2tap_qpel_pixels_tab));

    memset(c->avg_2tap_qpel_pixels_tab, 0, sizeof(c->avg_2tap_qpel_pixels_tab));



#undef FUNC

#undef FUNCC

#define FUNC(f, depth) f ## _ ## depth

#define FUNCC(f, depth) f ## _ ## depth ## _c



#define dspfunc1(PFX, IDX, NUM, depth)\

    c->PFX ## _pixels_tab[IDX][0] = FUNCC(PFX ## _pixels ## NUM        , depth);\

    c->PFX ## _pixels_tab[IDX][1] = FUNCC(PFX ## _pixels ## NUM ## _x2 , depth);\

    c->PFX ## _pixels_tab[IDX][2] = FUNCC(PFX ## _pixels ## NUM ## _y2 , depth);\

    c->PFX ## _pixels_tab[IDX][3] = FUNCC(PFX ## _pixels ## NUM ## _xy2, depth)



#define dspfunc2(PFX, IDX, NUM, depth)\

    c->PFX ## _pixels_tab[IDX][ 0] = FUNCC(PFX ## NUM ## _mc00, depth);\

    c->PFX ## _pixels_tab[IDX][ 1] = FUNCC(PFX ## NUM ## _mc10, depth);\

    c->PFX ## _pixels_tab[IDX][ 2] = FUNCC(PFX ## NUM ## _mc20, depth);\

    c->PFX ## _pixels_tab[IDX][ 3] = FUNCC(PFX ## NUM ## _mc30, depth);\

    c->PFX ## _pixels_tab[IDX][ 4] = FUNCC(PFX ## NUM ## _mc01, depth);\

    c->PFX ## _pixels_tab[IDX][ 5] = FUNCC(PFX ## NUM ## _mc11, depth);\

    c->PFX ## _pixels_tab[IDX][ 6] = FUNCC(PFX ## NUM ## _mc21, depth);\

    c->PFX ## _pixels_tab[IDX][ 7] = FUNCC(PFX ## NUM ## _mc31, depth);\

    c->PFX ## _pixels_tab[IDX][ 8] = FUNCC(PFX ## NUM ## _mc02, depth);\

    c->PFX ## _pixels_tab[IDX][ 9] = FUNCC(PFX ## NUM ## _mc12, depth);\

    c->PFX ## _pixels_tab[IDX][10] = FUNCC(PFX ## NUM ## _mc22, depth);\

    c->PFX ## _pixels_tab[IDX][11] = FUNCC(PFX ## NUM ## _mc32, depth);\

    c->PFX ## _pixels_tab[IDX][12] = FUNCC(PFX ## NUM ## _mc03, depth);\

    c->PFX ## _pixels_tab[IDX][13] = FUNCC(PFX ## NUM ## _mc13, depth);\

    c->PFX ## _pixels_tab[IDX][14] = FUNCC(PFX ## NUM ## _mc23, depth);\

    c->PFX ## _pixels_tab[IDX][15] = FUNCC(PFX ## NUM ## _mc33, depth)





#define BIT_DEPTH_FUNCS(depth, dct)\

    c->get_pixels                    = FUNCC(get_pixels   ## dct   , depth);\

    c->draw_edges                    = FUNCC(draw_edges            , depth);\

    c->emulated_edge_mc              = FUNC (ff_emulated_edge_mc   , depth);\

    c->clear_block                   = FUNCC(clear_block  ## dct   , depth);\

    c->clear_blocks                  = FUNCC(clear_blocks ## dct   , depth);\

    c->add_pixels8                   = FUNCC(add_pixels8  ## dct   , depth);\

    c->add_pixels4                   = FUNCC(add_pixels4  ## dct   , depth);\

    c->put_no_rnd_pixels_l2[0]       = FUNCC(put_no_rnd_pixels16_l2, depth);\

    c->put_no_rnd_pixels_l2[1]       = FUNCC(put_no_rnd_pixels8_l2 , depth);\

\

    c->put_h264_chroma_pixels_tab[0] = FUNCC(put_h264_chroma_mc8   , depth);\

    c->put_h264_chroma_pixels_tab[1] = FUNCC(put_h264_chroma_mc4   , depth);\

    c->put_h264_chroma_pixels_tab[2] = FUNCC(put_h264_chroma_mc2   , depth);\

    c->avg_h264_chroma_pixels_tab[0] = FUNCC(avg_h264_chroma_mc8   , depth);\

    c->avg_h264_chroma_pixels_tab[1] = FUNCC(avg_h264_chroma_mc4   , depth);\

    c->avg_h264_chroma_pixels_tab[2] = FUNCC(avg_h264_chroma_mc2   , depth);\

\

    dspfunc1(put       , 0, 16, depth);\

    dspfunc1(put       , 1,  8, depth);\

    dspfunc1(put       , 2,  4, depth);\

    dspfunc1(put       , 3,  2, depth);\

    dspfunc1(put_no_rnd, 0, 16, depth);\

    dspfunc1(put_no_rnd, 1,  8, depth);\

    dspfunc1(avg       , 0, 16, depth);\

    dspfunc1(avg       , 1,  8, depth);\

    dspfunc1(avg       , 2,  4, depth);\

    dspfunc1(avg       , 3,  2, depth);\

    dspfunc1(avg_no_rnd, 0, 16, depth);\

    dspfunc1(avg_no_rnd, 1,  8, depth);\

\

    dspfunc2(put_h264_qpel, 0, 16, depth);\

    dspfunc2(put_h264_qpel, 1,  8, depth);\

    dspfunc2(put_h264_qpel, 2,  4, depth);\

    dspfunc2(put_h264_qpel, 3,  2, depth);\

    dspfunc2(avg_h264_qpel, 0, 16, depth);\

    dspfunc2(avg_h264_qpel, 1,  8, depth);\

    dspfunc2(avg_h264_qpel, 2,  4, depth);



    switch (avctx->bits_per_raw_sample) {

    case 9:

        if (c->dct_bits == 32) {

            BIT_DEPTH_FUNCS(9, _32);

        } else {

            BIT_DEPTH_FUNCS(9, _16);

        }

        break;

    case 10:

        if (c->dct_bits == 32) {

            BIT_DEPTH_FUNCS(10, _32);

        } else {

            BIT_DEPTH_FUNCS(10, _16);

        }

        break;

    default:

        BIT_DEPTH_FUNCS(8, _16);

        break;

    }





    if (HAVE_MMX)        ff_dsputil_init_mmx   (c, avctx);

    if (ARCH_ARM)        ff_dsputil_init_arm   (c, avctx);

    if (HAVE_VIS)        ff_dsputil_init_vis   (c, avctx);

    if (ARCH_ALPHA)      ff_dsputil_init_alpha (c, avctx);

    if (ARCH_PPC)        ff_dsputil_init_ppc   (c, avctx);

    if (HAVE_MMI)        ff_dsputil_init_mmi   (c, avctx);

    if (ARCH_SH4)        ff_dsputil_init_sh4   (c, avctx);

    if (ARCH_BFIN)       ff_dsputil_init_bfin  (c, avctx);



    for(i=0; i<64; i++){

        if(!c->put_2tap_qpel_pixels_tab[0][i])

            c->put_2tap_qpel_pixels_tab[0][i]= c->put_h264_qpel_pixels_tab[0][i];

        if(!c->avg_2tap_qpel_pixels_tab[0][i])

            c->avg_2tap_qpel_pixels_tab[0][i]= c->avg_h264_qpel_pixels_tab[0][i];

    }



    ff_init_scantable_permutation(c->idct_permutation,

                                  c->idct_permutation_type);

}
