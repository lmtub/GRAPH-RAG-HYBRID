av_cold int ff_cavs_init(AVCodecContext *avctx)

{

    AVSContext *h = avctx->priv_data;



    ff_blockdsp_init(&h->bdsp, avctx);

    ff_h264chroma_init(&h->h264chroma, 8);

    ff_idctdsp_init(&h->idsp, avctx);

    ff_videodsp_init(&h->vdsp, 8);

    ff_cavsdsp_init(&h->cdsp, avctx);

    ff_init_scantable_permutation(h->idsp.idct_permutation,

                                  h->cdsp.idct_perm);

    ff_init_scantable(h->idsp.idct_permutation, &h->scantable, ff_zigzag_direct);



    h->avctx       = avctx;

    avctx->pix_fmt = AV_PIX_FMT_YUV420P;



    h->cur.f    = av_frame_alloc();

    h->DPB[0].f = av_frame_alloc();

    h->DPB[1].f = av_frame_alloc();

    if (!h->cur.f || !h->DPB[0].f || !h->DPB[1].f) {

        ff_cavs_end(avctx);

        return AVERROR(ENOMEM);

    }



    h->luma_scan[0]                     = 0;

    h->luma_scan[1]                     = 8;

    h->intra_pred_l[INTRA_L_VERT]       = intra_pred_vert;

    h->intra_pred_l[INTRA_L_HORIZ]      = intra_pred_horiz;

    h->intra_pred_l[INTRA_L_LP]         = intra_pred_lp;

    h->intra_pred_l[INTRA_L_DOWN_LEFT]  = intra_pred_down_left;

    h->intra_pred_l[INTRA_L_DOWN_RIGHT] = intra_pred_down_right;

    h->intra_pred_l[INTRA_L_LP_LEFT]    = intra_pred_lp_left;

    h->intra_pred_l[INTRA_L_LP_TOP]     = intra_pred_lp_top;

    h->intra_pred_l[INTRA_L_DC_128]     = intra_pred_dc_128;

    h->intra_pred_c[INTRA_C_LP]         = intra_pred_lp;

    h->intra_pred_c[INTRA_C_HORIZ]      = intra_pred_horiz;

    h->intra_pred_c[INTRA_C_VERT]       = intra_pred_vert;

    h->intra_pred_c[INTRA_C_PLANE]      = intra_pred_plane;

    h->intra_pred_c[INTRA_C_LP_LEFT]    = intra_pred_lp_left;

    h->intra_pred_c[INTRA_C_LP_TOP]     = intra_pred_lp_top;

    h->intra_pred_c[INTRA_C_DC_128]     = intra_pred_dc_128;

    h->mv[7]                            = un_mv;

    h->mv[19]                           = un_mv;

    return 0;

}
