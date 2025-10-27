static void fill_slice_long(AVCodecContext *avctx, DXVA_Slice_H264_Long *slice,

                            const DXVA_PicParams_H264 *pp, unsigned position, unsigned size)

{

    const H264Context *h = avctx->priv_data;

    H264SliceContext *sl = &h->slice_ctx[0];

    AVDXVAContext *ctx = avctx->hwaccel_context;

    unsigned list;



    memset(slice, 0, sizeof(*slice));

    slice->BSNALunitDataLocation = position;

    slice->SliceBytesInBuffer    = size;

    slice->wBadSliceChopping     = 0;



    slice->first_mb_in_slice     = (sl->mb_y >> FIELD_OR_MBAFF_PICTURE(h)) * h->mb_width + sl->mb_x;

    slice->NumMbsForSlice        = 0; /* XXX it is set once we have all slices */

    slice->BitOffsetToSliceData  = get_bits_count(&sl->gb) - 8;

    slice->slice_type            = ff_h264_get_slice_type(sl);

    if (sl->slice_type_fixed)

        slice->slice_type += 5;

    slice->luma_log2_weight_denom       = sl->pwt.luma_log2_weight_denom;

    slice->chroma_log2_weight_denom     = sl->pwt.chroma_log2_weight_denom;

    if (sl->list_count > 0)

        slice->num_ref_idx_l0_active_minus1 = sl->ref_count[0] - 1;

    if (sl->list_count > 1)

        slice->num_ref_idx_l1_active_minus1 = sl->ref_count[1] - 1;

    slice->slice_alpha_c0_offset_div2   = sl->slice_alpha_c0_offset / 2;

    slice->slice_beta_offset_div2       = sl->slice_beta_offset     / 2;

    slice->Reserved8Bits                = 0;



    for (list = 0; list < 2; list++) {

        unsigned i;

        for (i = 0; i < FF_ARRAY_ELEMS(slice->RefPicList[list]); i++) {

            if (list < sl->list_count && i < sl->ref_count[list]) {

                const H264Picture *r = sl->ref_list[list][i].parent;

                unsigned plane;

                unsigned index;

                if (DXVA_CONTEXT_WORKAROUND(avctx, ctx) & FF_DXVA2_WORKAROUND_INTEL_CLEARVIDEO)

                    index = ff_dxva2_get_surface_index(avctx, ctx, r->f);

                else

                    index = get_refpic_index(pp, ff_dxva2_get_surface_index(avctx, ctx, r->f));

                fill_picture_entry(&slice->RefPicList[list][i], index,

                                   sl->ref_list[list][i].reference == PICT_BOTTOM_FIELD);

                for (plane = 0; plane < 3; plane++) {

                    int w, o;

                    if (plane == 0 && sl->pwt.luma_weight_flag[list]) {

                        w = sl->pwt.luma_weight[i][list][0];

                        o = sl->pwt.luma_weight[i][list][1];

                    } else if (plane >= 1 && sl->pwt.chroma_weight_flag[list]) {

                        w = sl->pwt.chroma_weight[i][list][plane-1][0];

                        o = sl->pwt.chroma_weight[i][list][plane-1][1];

                    } else {

                        w = 1 << (plane == 0 ? sl->pwt.luma_log2_weight_denom :

                                               sl->pwt.chroma_log2_weight_denom);

                        o = 0;

                    }

                    slice->Weights[list][i][plane][0] = w;

                    slice->Weights[list][i][plane][1] = o;

                }

            } else {

                unsigned plane;

                slice->RefPicList[list][i].bPicEntry = 0xff;

                for (plane = 0; plane < 3; plane++) {

                    slice->Weights[list][i][plane][0] = 0;

                    slice->Weights[list][i][plane][1] = 0;

                }

            }

        }

    }

    slice->slice_qs_delta    = 0; /* XXX not implemented by FFmpeg */

    slice->slice_qp_delta    = sl->qscale - h->ps.pps->init_qp;

    slice->redundant_pic_cnt = sl->redundant_pic_count;

    if (sl->slice_type == AV_PICTURE_TYPE_B)

        slice->direct_spatial_mv_pred_flag = sl->direct_spatial_mv_pred;

    slice->cabac_init_idc = h->ps.pps->cabac ? sl->cabac_init_idc : 0;

    if (sl->deblocking_filter < 2)

        slice->disable_deblocking_filter_idc = 1 - sl->deblocking_filter;

    else

        slice->disable_deblocking_filter_idc = sl->deblocking_filter;

    slice->slice_id = h->current_slice - 1;

}
