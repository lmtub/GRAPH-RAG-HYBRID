static inline int decode_vui_parameters(H264Context *h, SPS *sps)

{

    int aspect_ratio_info_present_flag;

    unsigned int aspect_ratio_idc;



    aspect_ratio_info_present_flag = get_bits1(&h->gb);



    if (aspect_ratio_info_present_flag) {

        aspect_ratio_idc = get_bits(&h->gb, 8);

        if (aspect_ratio_idc == EXTENDED_SAR) {

            sps->sar.num = get_bits(&h->gb, 16);

            sps->sar.den = get_bits(&h->gb, 16);

        } else if (aspect_ratio_idc < FF_ARRAY_ELEMS(pixel_aspect)) {

            sps->sar = pixel_aspect[aspect_ratio_idc];

        } else {

            av_log(h->avctx, AV_LOG_ERROR, "illegal aspect ratio\n");

            return AVERROR_INVALIDDATA;

        }

    } else {

        sps->sar.num =

        sps->sar.den = 0;

    }



    if (get_bits1(&h->gb))      /* overscan_info_present_flag */

        get_bits1(&h->gb);      /* overscan_appropriate_flag */



    sps->video_signal_type_present_flag = get_bits1(&h->gb);

    if (sps->video_signal_type_present_flag) {

        get_bits(&h->gb, 3);                 /* video_format */

        sps->full_range = get_bits1(&h->gb); /* video_full_range_flag */



        sps->colour_description_present_flag = get_bits1(&h->gb);

        if (sps->colour_description_present_flag) {

            sps->color_primaries = get_bits(&h->gb, 8); /* colour_primaries */

            sps->color_trc       = get_bits(&h->gb, 8); /* transfer_characteristics */

            sps->colorspace      = get_bits(&h->gb, 8); /* matrix_coefficients */

            if (sps->color_primaries >= AVCOL_PRI_NB)

                sps->color_primaries = AVCOL_PRI_UNSPECIFIED;

            if (sps->color_trc >= AVCOL_TRC_NB)

                sps->color_trc = AVCOL_TRC_UNSPECIFIED;

            if (sps->colorspace >= AVCOL_SPC_NB)

                sps->colorspace = AVCOL_SPC_UNSPECIFIED;

        }

    }



    /* chroma_location_info_present_flag */

    if (get_bits1(&h->gb)) {

        /* chroma_sample_location_type_top_field */

        h->avctx->chroma_sample_location = get_ue_golomb(&h->gb) + 1;

        get_ue_golomb(&h->gb);  /* chroma_sample_location_type_bottom_field */

    }



    sps->timing_info_present_flag = get_bits1(&h->gb);

    if (sps->timing_info_present_flag) {

        sps->num_units_in_tick = get_bits_long(&h->gb, 32);

        sps->time_scale        = get_bits_long(&h->gb, 32);

        if (!sps->num_units_in_tick || !sps->time_scale) {

            av_log(h->avctx, AV_LOG_ERROR,

                   "time_scale/num_units_in_tick invalid or unsupported (%"PRIu32"/%"PRIu32")\n",

                   sps->time_scale, sps->num_units_in_tick);

            return AVERROR_INVALIDDATA;

        }

        sps->fixed_frame_rate_flag = get_bits1(&h->gb);

    }



    sps->nal_hrd_parameters_present_flag = get_bits1(&h->gb);

    if (sps->nal_hrd_parameters_present_flag)

        if (decode_hrd_parameters(h, sps) < 0)

            return AVERROR_INVALIDDATA;

    sps->vcl_hrd_parameters_present_flag = get_bits1(&h->gb);

    if (sps->vcl_hrd_parameters_present_flag)

        if (decode_hrd_parameters(h, sps) < 0)

            return AVERROR_INVALIDDATA;

    if (sps->nal_hrd_parameters_present_flag ||

        sps->vcl_hrd_parameters_present_flag)

        get_bits1(&h->gb);     /* low_delay_hrd_flag */

    sps->pic_struct_present_flag = get_bits1(&h->gb);



    sps->bitstream_restriction_flag = get_bits1(&h->gb);

    if (sps->bitstream_restriction_flag) {

        get_bits1(&h->gb);     /* motion_vectors_over_pic_boundaries_flag */

        get_ue_golomb(&h->gb); /* max_bytes_per_pic_denom */

        get_ue_golomb(&h->gb); /* max_bits_per_mb_denom */

        get_ue_golomb(&h->gb); /* log2_max_mv_length_horizontal */

        get_ue_golomb(&h->gb); /* log2_max_mv_length_vertical */

        sps->num_reorder_frames = get_ue_golomb(&h->gb);

        get_ue_golomb(&h->gb); /*max_dec_frame_buffering*/



        if (get_bits_left(&h->gb) < 0) {

            sps->num_reorder_frames         = 0;

            sps->bitstream_restriction_flag = 0;

        }



        if (sps->num_reorder_frames > 16U

            /* max_dec_frame_buffering || max_dec_frame_buffering > 16 */) {

            av_log(h->avctx, AV_LOG_ERROR,

                   "Clipping illegal num_reorder_frames %d\n",

                   sps->num_reorder_frames);

            sps->num_reorder_frames = 16;

            return AVERROR_INVALIDDATA;

        }

    }

    if (get_bits_left(&h->gb) < 0) {

        av_log(h->avctx, AV_LOG_ERROR,

               "Overread VUI by %d bits\n", -get_bits_left(&h->gb));

        return AVERROR_INVALIDDATA;

    }



    return 0;

}
