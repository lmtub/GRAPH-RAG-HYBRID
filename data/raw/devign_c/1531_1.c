static inline int decode_vui_parameters(H264Context *h, SPS *sps){

    MpegEncContext * const s = &h->s;

    int aspect_ratio_info_present_flag;

    unsigned int aspect_ratio_idc;



    aspect_ratio_info_present_flag= get_bits1(&s->gb);



    if( aspect_ratio_info_present_flag ) {

        aspect_ratio_idc= get_bits(&s->gb, 8);

        if( aspect_ratio_idc == EXTENDED_SAR ) {

            sps->sar.num= get_bits(&s->gb, 16);

            sps->sar.den= get_bits(&s->gb, 16);

        }else if(aspect_ratio_idc < FF_ARRAY_ELEMS(pixel_aspect)){

            sps->sar=  pixel_aspect[aspect_ratio_idc];

        }else{

            av_log(h->s.avctx, AV_LOG_ERROR, "illegal aspect ratio\n");

            return -1;

        }

    }else{

        sps->sar.num=

        sps->sar.den= 0;

    }

//            s->avctx->aspect_ratio= sar_width*s->width / (float)(s->height*sar_height);



    if(get_bits1(&s->gb)){      /* overscan_info_present_flag */

        get_bits1(&s->gb);      /* overscan_appropriate_flag */

    }



    sps->video_signal_type_present_flag = get_bits1(&s->gb);

    if(sps->video_signal_type_present_flag){

        get_bits(&s->gb, 3);    /* video_format */

        sps->full_range = get_bits1(&s->gb); /* video_full_range_flag */



        sps->colour_description_present_flag = get_bits1(&s->gb);

        if(sps->colour_description_present_flag){

            sps->color_primaries = get_bits(&s->gb, 8); /* colour_primaries */

            sps->color_trc       = get_bits(&s->gb, 8); /* transfer_characteristics */

            sps->colorspace      = get_bits(&s->gb, 8); /* matrix_coefficients */

            if (sps->color_primaries >= AVCOL_PRI_NB)

                sps->color_primaries  = AVCOL_PRI_UNSPECIFIED;

            if (sps->color_trc >= AVCOL_TRC_NB)

                sps->color_trc  = AVCOL_TRC_UNSPECIFIED;

            if (sps->colorspace >= AVCOL_SPC_NB)

                sps->colorspace  = AVCOL_SPC_UNSPECIFIED;

        }

    }



    if(get_bits1(&s->gb)){      /* chroma_location_info_present_flag */

        s->avctx->chroma_sample_location = get_ue_golomb(&s->gb)+1;  /* chroma_sample_location_type_top_field */

        get_ue_golomb(&s->gb);  /* chroma_sample_location_type_bottom_field */

    }



    sps->timing_info_present_flag = get_bits1(&s->gb);

    if(sps->timing_info_present_flag){

        sps->num_units_in_tick = get_bits_long(&s->gb, 32);

        sps->time_scale = get_bits_long(&s->gb, 32);

        if(!sps->num_units_in_tick || !sps->time_scale){

            av_log(h->s.avctx, AV_LOG_ERROR, "time_scale/num_units_in_tick invalid or unsupported (%d/%d)\n", sps->time_scale, sps->num_units_in_tick);

            return -1;

        }

        sps->fixed_frame_rate_flag = get_bits1(&s->gb);

    }



    sps->nal_hrd_parameters_present_flag = get_bits1(&s->gb);

    if(sps->nal_hrd_parameters_present_flag)

        if(decode_hrd_parameters(h, sps) < 0)

            return -1;

    sps->vcl_hrd_parameters_present_flag = get_bits1(&s->gb);

    if(sps->vcl_hrd_parameters_present_flag)

        if(decode_hrd_parameters(h, sps) < 0)

            return -1;

    if(sps->nal_hrd_parameters_present_flag || sps->vcl_hrd_parameters_present_flag)

        get_bits1(&s->gb);     /* low_delay_hrd_flag */

    sps->pic_struct_present_flag = get_bits1(&s->gb);



    sps->bitstream_restriction_flag = get_bits1(&s->gb);

    if(sps->bitstream_restriction_flag){

        get_bits1(&s->gb);     /* motion_vectors_over_pic_boundaries_flag */

        get_ue_golomb(&s->gb); /* max_bytes_per_pic_denom */

        get_ue_golomb(&s->gb); /* max_bits_per_mb_denom */

        get_ue_golomb(&s->gb); /* log2_max_mv_length_horizontal */

        get_ue_golomb(&s->gb); /* log2_max_mv_length_vertical */

        sps->num_reorder_frames= get_ue_golomb(&s->gb);

        get_ue_golomb(&s->gb); /*max_dec_frame_buffering*/



        if (get_bits_left(&s->gb) < 0) {

            av_log(h->s.avctx, AV_LOG_ERROR, "Overread VUI by %d bits\n", -get_bits_left(&s->gb));

            sps->num_reorder_frames=0;

            sps->bitstream_restriction_flag= 0;

        }



        if(sps->num_reorder_frames > 16U /*max_dec_frame_buffering || max_dec_frame_buffering > 16*/){

            av_log(h->s.avctx, AV_LOG_ERROR, "illegal num_reorder_frames %d\n", sps->num_reorder_frames);

            return -1;

        }

    }



    return 0;

}
