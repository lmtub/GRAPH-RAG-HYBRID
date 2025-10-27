static inline int decode_seq_parameter_set(H264Context *h){

    MpegEncContext * const s = &h->s;

    int profile_idc, level_idc;

    int sps_id, i;

    SPS *sps;



    profile_idc= get_bits(&s->gb, 8);

    get_bits1(&s->gb);   //constraint_set0_flag

    get_bits1(&s->gb);   //constraint_set1_flag

    get_bits1(&s->gb);   //constraint_set2_flag

    get_bits1(&s->gb);   //constraint_set3_flag

    get_bits(&s->gb, 4); // reserved

    level_idc= get_bits(&s->gb, 8);

    sps_id= get_ue_golomb(&s->gb);



    sps= &h->sps_buffer[ sps_id ];

    sps->profile_idc= profile_idc;

    sps->level_idc= level_idc;



    if(sps->profile_idc >= 100){ //high profile

        if(get_ue_golomb(&s->gb) == 3) //chroma_format_idc

            get_bits1(&s->gb);  //residual_color_transform_flag

        get_ue_golomb(&s->gb);  //bit_depth_luma_minus8

        get_ue_golomb(&s->gb);  //bit_depth_chroma_minus8

        sps->transform_bypass = get_bits1(&s->gb);

        decode_scaling_matrices(h, sps, NULL, 1, sps->scaling_matrix4, sps->scaling_matrix8);

    }else

        sps->scaling_matrix_present = 0;



    sps->log2_max_frame_num= get_ue_golomb(&s->gb) + 4;

    sps->poc_type= get_ue_golomb(&s->gb);



    if(sps->poc_type == 0){ //FIXME #define

        sps->log2_max_poc_lsb= get_ue_golomb(&s->gb) + 4;

    } else if(sps->poc_type == 1){//FIXME #define

        sps->delta_pic_order_always_zero_flag= get_bits1(&s->gb);

        sps->offset_for_non_ref_pic= get_se_golomb(&s->gb);

        sps->offset_for_top_to_bottom_field= get_se_golomb(&s->gb);

        sps->poc_cycle_length= get_ue_golomb(&s->gb);



        for(i=0; i<sps->poc_cycle_length; i++)

            sps->offset_for_ref_frame[i]= get_se_golomb(&s->gb);

    }

    if(sps->poc_type > 2){

        av_log(h->s.avctx, AV_LOG_ERROR, "illegal POC type %d\n", sps->poc_type);

        return -1;

    }



    sps->ref_frame_count= get_ue_golomb(&s->gb);

    if(sps->ref_frame_count > MAX_PICTURE_COUNT-2){

        av_log(h->s.avctx, AV_LOG_ERROR, "too many reference frames\n");

    }

    sps->gaps_in_frame_num_allowed_flag= get_bits1(&s->gb);

    sps->mb_width= get_ue_golomb(&s->gb) + 1;

    sps->mb_height= get_ue_golomb(&s->gb) + 1;

    if((unsigned)sps->mb_width >= INT_MAX/16 || (unsigned)sps->mb_height >= INT_MAX/16 ||

       avcodec_check_dimensions(NULL, 16*sps->mb_width, 16*sps->mb_height))

        return -1;



    sps->frame_mbs_only_flag= get_bits1(&s->gb);

    if(!sps->frame_mbs_only_flag)

        sps->mb_aff= get_bits1(&s->gb);

    else

        sps->mb_aff= 0;



    sps->direct_8x8_inference_flag= get_bits1(&s->gb);



#ifndef ALLOW_INTERLACE

    if(sps->mb_aff)

        av_log(h->s.avctx, AV_LOG_ERROR, "MBAFF support not included; enable it at compile-time.\n");

#endif

    if(!sps->direct_8x8_inference_flag && sps->mb_aff)

        av_log(h->s.avctx, AV_LOG_ERROR, "MBAFF + !direct_8x8_inference is not implemented\n");



    sps->crop= get_bits1(&s->gb);

    if(sps->crop){

        sps->crop_left  = get_ue_golomb(&s->gb);

        sps->crop_right = get_ue_golomb(&s->gb);

        sps->crop_top   = get_ue_golomb(&s->gb);

        sps->crop_bottom= get_ue_golomb(&s->gb);

        if(sps->crop_left || sps->crop_top){

            av_log(h->s.avctx, AV_LOG_ERROR, "insane cropping not completely supported, this could look slightly wrong ...\n");

        }

    }else{

        sps->crop_left  =

        sps->crop_right =

        sps->crop_top   =

        sps->crop_bottom= 0;

    }



    sps->vui_parameters_present_flag= get_bits1(&s->gb);

    if( sps->vui_parameters_present_flag )

        decode_vui_parameters(h, sps);



    if(s->avctx->debug&FF_DEBUG_PICT_INFO){

        av_log(h->s.avctx, AV_LOG_DEBUG, "sps:%d profile:%d/%d poc:%d ref:%d %dx%d %s %s crop:%d/%d/%d/%d %s\n",

               sps_id, sps->profile_idc, sps->level_idc,

               sps->poc_type,

               sps->ref_frame_count,

               sps->mb_width, sps->mb_height,

               sps->frame_mbs_only_flag ? "FRM" : (sps->mb_aff ? "MB-AFF" : "PIC-AFF"),

               sps->direct_8x8_inference_flag ? "8B8" : "",

               sps->crop_left, sps->crop_right,

               sps->crop_top, sps->crop_bottom,

               sps->vui_parameters_present_flag ? "VUI" : ""

               );

    }

    return 0;

}
