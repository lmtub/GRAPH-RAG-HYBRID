static int vaapi_h264_decode_slice(AVCodecContext *avctx,

                                   const uint8_t  *buffer,

                                   uint32_t        size)

{

    H264Context * const h = avctx->priv_data;

    H264SliceContext *sl  = &h->slice_ctx[0];

    VASliceParameterBufferH264 *slice_param;



    ff_dlog(avctx, "vaapi_h264_decode_slice(): buffer %p, size %d\n",

            buffer, size);



    /* Fill in VASliceParameterBufferH264. */

    slice_param = (VASliceParameterBufferH264 *)ff_vaapi_alloc_slice(avctx->hwaccel_context, buffer, size);

    if (!slice_param)

        return -1;

    slice_param->slice_data_bit_offset          = get_bits_count(&sl->gb);

    slice_param->first_mb_in_slice              = (sl->mb_y >> FIELD_OR_MBAFF_PICTURE(h)) * h->mb_width + sl->mb_x;

    slice_param->slice_type                     = ff_h264_get_slice_type(sl);

    slice_param->direct_spatial_mv_pred_flag    = sl->slice_type == AV_PICTURE_TYPE_B ? sl->direct_spatial_mv_pred : 0;

    slice_param->num_ref_idx_l0_active_minus1   = sl->list_count > 0 ? sl->ref_count[0] - 1 : 0;

    slice_param->num_ref_idx_l1_active_minus1   = sl->list_count > 1 ? sl->ref_count[1] - 1 : 0;

    slice_param->cabac_init_idc                 = sl->cabac_init_idc;

    slice_param->slice_qp_delta                 = sl->qscale - h->pps.init_qp;

    slice_param->disable_deblocking_filter_idc  = sl->deblocking_filter < 2 ? !sl->deblocking_filter : sl->deblocking_filter;

    slice_param->slice_alpha_c0_offset_div2     = sl->slice_alpha_c0_offset / 2;

    slice_param->slice_beta_offset_div2         = sl->slice_beta_offset     / 2;

    slice_param->luma_log2_weight_denom         = sl->pwt.luma_log2_weight_denom;

    slice_param->chroma_log2_weight_denom       = sl->pwt.chroma_log2_weight_denom;



    fill_vaapi_RefPicList(slice_param->RefPicList0, sl->ref_list[0], sl->list_count > 0 ? sl->ref_count[0] : 0);

    fill_vaapi_RefPicList(slice_param->RefPicList1, sl->ref_list[1], sl->list_count > 1 ? sl->ref_count[1] : 0);



    fill_vaapi_plain_pred_weight_table(h, 0,

                                       &slice_param->luma_weight_l0_flag,   slice_param->luma_weight_l0,   slice_param->luma_offset_l0,

                                       &slice_param->chroma_weight_l0_flag, slice_param->chroma_weight_l0, slice_param->chroma_offset_l0);

    fill_vaapi_plain_pred_weight_table(h, 1,

                                       &slice_param->luma_weight_l1_flag,   slice_param->luma_weight_l1,   slice_param->luma_offset_l1,

                                       &slice_param->chroma_weight_l1_flag, slice_param->chroma_weight_l1, slice_param->chroma_offset_l1);

    return 0;

}
