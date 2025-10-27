static int vaapi_encode_h264_init_slice_params(AVCodecContext *avctx,

                                               VAAPIEncodePicture *pic,

                                               VAAPIEncodeSlice *slice)

{

    VAAPIEncodeContext                 *ctx = avctx->priv_data;

    VAEncSequenceParameterBufferH264  *vseq = ctx->codec_sequence_params;

    VAEncPictureParameterBufferH264   *vpic = pic->codec_picture_params;

    VAEncSliceParameterBufferH264   *vslice = slice->codec_slice_params;

    VAAPIEncodeH264Context            *priv = ctx->priv_data;

    VAAPIEncodeH264Slice            *pslice;

    VAAPIEncodeH264MiscSliceParams  *mslice;

    int i;



    slice->priv_data = av_mallocz(sizeof(*pslice));

    if (!slice->priv_data)

        return AVERROR(ENOMEM);

    pslice = slice->priv_data;

    mslice = &pslice->misc_slice_params;



    if (pic->type == PICTURE_TYPE_IDR)

        mslice->nal_unit_type = H264_NAL_IDR_SLICE;

    else

        mslice->nal_unit_type = H264_NAL_SLICE;



    switch (pic->type) {

    case PICTURE_TYPE_IDR:

        vslice->slice_type  = SLICE_TYPE_I;

        mslice->nal_ref_idc = 3;

        break;

    case PICTURE_TYPE_I:

        vslice->slice_type  = SLICE_TYPE_I;

        mslice->nal_ref_idc = 2;

        break;

    case PICTURE_TYPE_P:

        vslice->slice_type  = SLICE_TYPE_P;

        mslice->nal_ref_idc = 1;

        break;

    case PICTURE_TYPE_B:

        vslice->slice_type  = SLICE_TYPE_B;

        mslice->nal_ref_idc = 0;

        break;

    default:

        av_assert0(0 && "invalid picture type");

    }



    // Only one slice per frame.

    vslice->macroblock_address = 0;

    vslice->num_macroblocks = priv->mb_width * priv->mb_height;



    vslice->macroblock_info = VA_INVALID_ID;



    vslice->pic_parameter_set_id = vpic->pic_parameter_set_id;

    vslice->idr_pic_id = priv->idr_pic_count++;



    vslice->pic_order_cnt_lsb = pic->display_order &

        ((1 << (4 + vseq->seq_fields.bits.log2_max_pic_order_cnt_lsb_minus4)) - 1);



    for (i = 0; i < FF_ARRAY_ELEMS(vslice->RefPicList0); i++) {

        vslice->RefPicList0[i].picture_id = VA_INVALID_ID;

        vslice->RefPicList0[i].flags      = VA_PICTURE_H264_INVALID;

        vslice->RefPicList1[i].picture_id = VA_INVALID_ID;

        vslice->RefPicList1[i].flags      = VA_PICTURE_H264_INVALID;

    }



    av_assert0(pic->nb_refs <= 2);

    if (pic->nb_refs >= 1) {

        // Backward reference for P- or B-frame.

        av_assert0(pic->type == PICTURE_TYPE_P ||

                   pic->type == PICTURE_TYPE_B);



        vslice->num_ref_idx_l0_active_minus1 = 0;

        vslice->RefPicList0[0] = vpic->ReferenceFrames[0];

    }

    if (pic->nb_refs >= 2) {

        // Forward reference for B-frame.

        av_assert0(pic->type == PICTURE_TYPE_B);



        vslice->num_ref_idx_l1_active_minus1 = 0;

        vslice->RefPicList1[0] = vpic->ReferenceFrames[1];

    }



    if (pic->type == PICTURE_TYPE_B)

        vslice->slice_qp_delta = priv->fixed_qp_b - vpic->pic_init_qp;

    else if (pic->type == PICTURE_TYPE_P)

        vslice->slice_qp_delta = priv->fixed_qp_p - vpic->pic_init_qp;

    else

        vslice->slice_qp_delta = priv->fixed_qp_idr - vpic->pic_init_qp;



    vslice->direct_spatial_mv_pred_flag = 1;



    return 0;

}
