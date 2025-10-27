static int vaapi_encode_h264_init_sequence_params(AVCodecContext *avctx)

{

    VAAPIEncodeContext                 *ctx = avctx->priv_data;

    VAEncSequenceParameterBufferH264  *vseq = ctx->codec_sequence_params;

    VAEncPictureParameterBufferH264   *vpic = ctx->codec_picture_params;

    VAAPIEncodeH264Context            *priv = ctx->priv_data;

    VAAPIEncodeH264MiscSequenceParams *mseq = &priv->misc_sequence_params;

    int i;



    {

        vseq->seq_parameter_set_id = 0;



        vseq->level_idc = avctx->level;



        vseq->max_num_ref_frames = 1 + (avctx->max_b_frames > 0);



        vseq->picture_width_in_mbs  = priv->mb_width;

        vseq->picture_height_in_mbs = priv->mb_height;



        vseq->seq_fields.bits.chroma_format_idc = 1;

        vseq->seq_fields.bits.frame_mbs_only_flag = 1;

        vseq->seq_fields.bits.direct_8x8_inference_flag = 1;

        vseq->seq_fields.bits.log2_max_frame_num_minus4 = 4;

        vseq->seq_fields.bits.pic_order_cnt_type = 0;





        if (avctx->width  != ctx->surface_width ||

            avctx->height != ctx->surface_height) {

            vseq->frame_cropping_flag = 1;



            vseq->frame_crop_left_offset   = 0;

            vseq->frame_crop_right_offset  =

                (ctx->surface_width - avctx->width) / 2;

            vseq->frame_crop_top_offset    = 0;

            vseq->frame_crop_bottom_offset =

                (ctx->surface_height - avctx->height) / 2;

        } else {

            vseq->frame_cropping_flag = 0;

        }



        vseq->vui_parameters_present_flag = 1;

        if (avctx->sample_aspect_ratio.num != 0) {

            vseq->vui_fields.bits.aspect_ratio_info_present_flag = 1;

            // There is a large enum of these which we could support

            // individually rather than using the generic X/Y form?

            if (avctx->sample_aspect_ratio.num ==

                avctx->sample_aspect_ratio.den) {

                vseq->aspect_ratio_idc = 1;

            } else {

                vseq->aspect_ratio_idc = 255; // Extended SAR.

                vseq->sar_width  = avctx->sample_aspect_ratio.num;

                vseq->sar_height = avctx->sample_aspect_ratio.den;

            }

        }

        if (avctx->color_primaries != AVCOL_PRI_UNSPECIFIED ||

            avctx->color_trc       != AVCOL_TRC_UNSPECIFIED ||

            avctx->colorspace      != AVCOL_SPC_UNSPECIFIED) {

            mseq->video_signal_type_present_flag = 1;

            mseq->video_format             = 5; // Unspecified.

            mseq->video_full_range_flag    = 0;

            mseq->colour_description_present_flag = 1;

            // These enums are derived from the standard and hence

            // we can just use the values directly.

            mseq->colour_primaries         = avctx->color_primaries;

            mseq->transfer_characteristics = avctx->color_trc;

            mseq->matrix_coefficients      = avctx->colorspace;

        }



        vseq->vui_fields.bits.bitstream_restriction_flag = 1;

        mseq->motion_vectors_over_pic_boundaries_flag = 1;

        mseq->max_bytes_per_pic_denom = 0;

        mseq->max_bits_per_mb_denom   = 0;

        vseq->vui_fields.bits.log2_max_mv_length_horizontal = 16;

        vseq->vui_fields.bits.log2_max_mv_length_vertical   = 16;



        mseq->max_num_reorder_frames = (avctx->max_b_frames > 0);

        mseq->max_dec_pic_buffering  = vseq->max_num_ref_frames;



        vseq->bits_per_second = avctx->bit_rate;



        vseq->vui_fields.bits.timing_info_present_flag = 1;

        if (avctx->framerate.num > 0 && avctx->framerate.den > 0) {

            vseq->num_units_in_tick = avctx->framerate.den;

            vseq->time_scale        = 2 * avctx->framerate.num;

            mseq->fixed_frame_rate_flag = 1;

        } else {

            vseq->num_units_in_tick = avctx->time_base.num;

            vseq->time_scale        = 2 * avctx->time_base.den;

            mseq->fixed_frame_rate_flag = 0;

        }



        if (ctx->va_rc_mode == VA_RC_CBR) {

            priv->send_timing_sei = 1;

            mseq->nal_hrd_parameters_present_flag = 1;



            mseq->cpb_cnt_minus1 = 0;



            // Try to scale these to a sensible range so that the

            // golomb encode of the value is not overlong.

            mseq->bit_rate_scale =

                av_clip_uintp2(av_log2(avctx->bit_rate) - 15 - 6, 4);

            mseq->bit_rate_value_minus1[0] =

                (avctx->bit_rate >> mseq->bit_rate_scale + 6) - 1;



            mseq->cpb_size_scale =

                av_clip_uintp2(av_log2(ctx->hrd_params.hrd.buffer_size) - 15 - 4, 4);

            mseq->cpb_size_value_minus1[0] =

                (ctx->hrd_params.hrd.buffer_size >> mseq->cpb_size_scale + 4) - 1;



            // CBR mode isn't actually available here, despite naming.

            mseq->cbr_flag[0] = 0;



            mseq->initial_cpb_removal_delay_length_minus1 = 23;

            mseq->cpb_removal_delay_length_minus1         = 23;

            mseq->dpb_output_delay_length_minus1          = 7;

            mseq->time_offset_length = 0;



            // This calculation can easily overflow 32 bits.

            mseq->initial_cpb_removal_delay = 90000 *

                (uint64_t)ctx->hrd_params.hrd.initial_buffer_fullness /

                ctx->hrd_params.hrd.buffer_size;



            mseq->initial_cpb_removal_delay_offset = 0;

        } else {

            priv->send_timing_sei = 0;

            mseq->nal_hrd_parameters_present_flag = 0;

        }



        vseq->intra_period     = ctx->p_per_i * (ctx->b_per_p + 1);

        vseq->intra_idr_period = vseq->intra_period;

        vseq->ip_period        = ctx->b_per_p + 1;

    }



    {

        vpic->CurrPic.picture_id = VA_INVALID_ID;

        vpic->CurrPic.flags      = VA_PICTURE_H264_INVALID;



        for (i = 0; i < FF_ARRAY_ELEMS(vpic->ReferenceFrames); i++) {

            vpic->ReferenceFrames[i].picture_id = VA_INVALID_ID;

            vpic->ReferenceFrames[i].flags      = VA_PICTURE_H264_INVALID;

        }



        vpic->coded_buf = VA_INVALID_ID;



        vpic->pic_parameter_set_id = 0;

        vpic->seq_parameter_set_id = 0;



        vpic->num_ref_idx_l0_active_minus1 = 0;

        vpic->num_ref_idx_l1_active_minus1 = 0;



        vpic->pic_fields.bits.entropy_coding_mode_flag =

            ((avctx->profile & 0xff) != 66);

        vpic->pic_fields.bits.weighted_pred_flag = 0;

        vpic->pic_fields.bits.weighted_bipred_idc = 0;

        vpic->pic_fields.bits.transform_8x8_mode_flag =

            ((avctx->profile & 0xff) >= 100);



        vpic->pic_init_qp = priv->fixed_qp_idr;

    }



    {

        mseq->profile_idc = avctx->profile & 0xff;



        if (avctx->profile & FF_PROFILE_H264_CONSTRAINED)

            mseq->constraint_set1_flag = 1;

        if (avctx->profile & FF_PROFILE_H264_INTRA)

            mseq->constraint_set3_flag = 1;

    }



    return 0;

}