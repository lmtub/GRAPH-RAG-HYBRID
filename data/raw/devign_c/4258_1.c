static int decode_vop_header(MpegEncContext *s, GetBitContext *gb){

    int time_incr, time_increment;



    s->pict_type = get_bits(gb, 2) + AV_PICTURE_TYPE_I;        /* pict type: I = 0 , P = 1 */

    if(s->pict_type==AV_PICTURE_TYPE_B && s->low_delay && s->vol_control_parameters==0 && !(s->flags & CODEC_FLAG_LOW_DELAY)){

        av_log(s->avctx, AV_LOG_ERROR, "low_delay flag incorrectly, clearing it\n");

        s->low_delay=0;

    }



    s->partitioned_frame= s->data_partitioning && s->pict_type!=AV_PICTURE_TYPE_B;

    if(s->partitioned_frame)

        s->decode_mb= mpeg4_decode_partitioned_mb;

    else

        s->decode_mb= mpeg4_decode_mb;



    time_incr=0;

    while (get_bits1(gb) != 0)

        time_incr++;



    check_marker(gb, "before time_increment");



    if(s->time_increment_bits==0 || !(show_bits(gb, s->time_increment_bits+1)&1)){

        av_log(s->avctx, AV_LOG_ERROR, "hmm, seems the headers are not complete, trying to guess time_increment_bits\n");



        for(s->time_increment_bits=1 ;s->time_increment_bits<16; s->time_increment_bits++){

            if (    s->pict_type == AV_PICTURE_TYPE_P

                || (s->pict_type == AV_PICTURE_TYPE_S && s->vol_sprite_usage==GMC_SPRITE)) {

                if((show_bits(gb, s->time_increment_bits+6)&0x37) == 0x30) break;

            }else

                if((show_bits(gb, s->time_increment_bits+5)&0x1F) == 0x18) break;

        }



        av_log(s->avctx, AV_LOG_ERROR, "my guess is %d bits ;)\n",s->time_increment_bits);

    }



    if(IS_3IV1) time_increment= get_bits1(gb); //FIXME investigate further

    else time_increment= get_bits(gb, s->time_increment_bits);



    if(s->pict_type!=AV_PICTURE_TYPE_B){

        s->last_time_base= s->time_base;

        s->time_base+= time_incr;

        s->time= s->time_base*s->avctx->time_base.den + time_increment;

        if(s->workaround_bugs&FF_BUG_UMP4){

            if(s->time < s->last_non_b_time){

                /* header is not mpeg-4-compatible, broken encoder,

                 * trying to workaround */

                s->time_base++;

                s->time+= s->avctx->time_base.den;

            }

        }

        s->pp_time= s->time - s->last_non_b_time;

        s->last_non_b_time= s->time;

    }else{

        s->time= (s->last_time_base + time_incr)*s->avctx->time_base.den + time_increment;

        s->pb_time= s->pp_time - (s->last_non_b_time - s->time);

        if(s->pp_time <=s->pb_time || s->pp_time <= s->pp_time - s->pb_time || s->pp_time<=0){

            /* messed up order, maybe after seeking? skipping current b-frame */

            return FRAME_SKIPPED;

        }

        ff_mpeg4_init_direct_mv(s);



        if(s->t_frame==0) s->t_frame= s->pb_time;

        if(s->t_frame==0) s->t_frame=1; // 1/0 protection

        s->pp_field_time= (  ROUNDED_DIV(s->last_non_b_time, s->t_frame)

                           - ROUNDED_DIV(s->last_non_b_time - s->pp_time, s->t_frame))*2;

        s->pb_field_time= (  ROUNDED_DIV(s->time, s->t_frame)

                           - ROUNDED_DIV(s->last_non_b_time - s->pp_time, s->t_frame))*2;

        if(!s->progressive_sequence){

            if(s->pp_field_time <= s->pb_field_time || s->pb_field_time <= 1)

                return FRAME_SKIPPED;

        }

    }



    if(s->avctx->time_base.num)

        s->current_picture_ptr->f.pts = (s->time + s->avctx->time_base.num / 2) / s->avctx->time_base.num;

    else

        s->current_picture_ptr->f.pts = AV_NOPTS_VALUE;

    if(s->avctx->debug&FF_DEBUG_PTS)

        av_log(s->avctx, AV_LOG_DEBUG, "MPEG4 PTS: %"PRId64"\n",

               s->current_picture_ptr->f.pts);



    check_marker(gb, "before vop_coded");



    /* vop coded */

    if (get_bits1(gb) != 1){

        if(s->avctx->debug&FF_DEBUG_PICT_INFO)

            av_log(s->avctx, AV_LOG_ERROR, "vop not coded\n");

        return FRAME_SKIPPED;

    }

    if (s->shape != BIN_ONLY_SHAPE && ( s->pict_type == AV_PICTURE_TYPE_P

                          || (s->pict_type == AV_PICTURE_TYPE_S && s->vol_sprite_usage==GMC_SPRITE))) {

        /* rounding type for motion estimation */

        s->no_rounding = get_bits1(gb);

    } else {

        s->no_rounding = 0;

    }

//FIXME reduced res stuff



     if (s->shape != RECT_SHAPE) {

         if (s->vol_sprite_usage != 1 || s->pict_type != AV_PICTURE_TYPE_I) {

             skip_bits(gb, 13); /* width */

             skip_bits1(gb);   /* marker */

             skip_bits(gb, 13); /* height */

             skip_bits1(gb);   /* marker */

             skip_bits(gb, 13); /* hor_spat_ref */

             skip_bits1(gb);   /* marker */

             skip_bits(gb, 13); /* ver_spat_ref */

         }

         skip_bits1(gb); /* change_CR_disable */



         if (get_bits1(gb) != 0) {

             skip_bits(gb, 8); /* constant_alpha_value */

         }

     }

//FIXME complexity estimation stuff



     if (s->shape != BIN_ONLY_SHAPE) {

         skip_bits_long(gb, s->cplx_estimation_trash_i);

         if(s->pict_type != AV_PICTURE_TYPE_I)

            skip_bits_long(gb, s->cplx_estimation_trash_p);

         if(s->pict_type == AV_PICTURE_TYPE_B)

            skip_bits_long(gb, s->cplx_estimation_trash_b);



         s->intra_dc_threshold= ff_mpeg4_dc_threshold[ get_bits(gb, 3) ];

         if(!s->progressive_sequence){

             s->top_field_first= get_bits1(gb);

             s->alternate_scan= get_bits1(gb);

         }else

             s->alternate_scan= 0;

     }



     if(s->alternate_scan){

         ff_init_scantable(s->dsp.idct_permutation, &s->inter_scantable  , ff_alternate_vertical_scan);

         ff_init_scantable(s->dsp.idct_permutation, &s->intra_scantable  , ff_alternate_vertical_scan);

         ff_init_scantable(s->dsp.idct_permutation, &s->intra_h_scantable, ff_alternate_vertical_scan);

         ff_init_scantable(s->dsp.idct_permutation, &s->intra_v_scantable, ff_alternate_vertical_scan);

     } else{

         ff_init_scantable(s->dsp.idct_permutation, &s->inter_scantable  , ff_zigzag_direct);

         ff_init_scantable(s->dsp.idct_permutation, &s->intra_scantable  , ff_zigzag_direct);

         ff_init_scantable(s->dsp.idct_permutation, &s->intra_h_scantable, ff_alternate_horizontal_scan);

         ff_init_scantable(s->dsp.idct_permutation, &s->intra_v_scantable, ff_alternate_vertical_scan);

     }



     if(s->pict_type == AV_PICTURE_TYPE_S && (s->vol_sprite_usage==STATIC_SPRITE || s->vol_sprite_usage==GMC_SPRITE)){

         mpeg4_decode_sprite_trajectory(s, gb);

         if(s->sprite_brightness_change) av_log(s->avctx, AV_LOG_ERROR, "sprite_brightness_change not supported\n");

         if(s->vol_sprite_usage==STATIC_SPRITE) av_log(s->avctx, AV_LOG_ERROR, "static sprite not supported\n");

     }



     if (s->shape != BIN_ONLY_SHAPE) {

         s->chroma_qscale= s->qscale = get_bits(gb, s->quant_precision);

         if(s->qscale==0){

             av_log(s->avctx, AV_LOG_ERROR, "Error, header damaged or not MPEG4 header (qscale=0)\n");

             return -1; // makes no sense to continue, as there is nothing left from the image then

         }



         if (s->pict_type != AV_PICTURE_TYPE_I) {

             s->f_code = get_bits(gb, 3);       /* fcode_for */

             if(s->f_code==0){

                 av_log(s->avctx, AV_LOG_ERROR, "Error, header damaged or not MPEG4 header (f_code=0)\n");

                 return -1; // makes no sense to continue, as the MV decoding will break very quickly

             }

         }else

             s->f_code=1;



         if (s->pict_type == AV_PICTURE_TYPE_B) {

             s->b_code = get_bits(gb, 3);

         }else

             s->b_code=1;



         if(s->avctx->debug&FF_DEBUG_PICT_INFO){

             av_log(s->avctx, AV_LOG_DEBUG, "qp:%d fc:%d,%d %s size:%d pro:%d alt:%d top:%d %spel part:%d resync:%d w:%d a:%d rnd:%d vot:%d%s dc:%d ce:%d/%d/%d\n",

                 s->qscale, s->f_code, s->b_code,

                 s->pict_type == AV_PICTURE_TYPE_I ? "I" : (s->pict_type == AV_PICTURE_TYPE_P ? "P" : (s->pict_type == AV_PICTURE_TYPE_B ? "B" : "S")),

                 gb->size_in_bits,s->progressive_sequence, s->alternate_scan, s->top_field_first,

                 s->quarter_sample ? "q" : "h", s->data_partitioning, s->resync_marker, s->num_sprite_warping_points,

                 s->sprite_warping_accuracy, 1-s->no_rounding, s->vo_type, s->vol_control_parameters ? " VOLC" : " ", s->intra_dc_threshold, s->cplx_estimation_trash_i, s->cplx_estimation_trash_p, s->cplx_estimation_trash_b);

         }



         if(!s->scalability){

             if (s->shape!=RECT_SHAPE && s->pict_type!=AV_PICTURE_TYPE_I) {

                 skip_bits1(gb); // vop shape coding type

             }

         }else{

             if(s->enhancement_type){

                 int load_backward_shape= get_bits1(gb);

                 if(load_backward_shape){

                     av_log(s->avctx, AV_LOG_ERROR, "load backward shape isn't supported\n");

                 }

             }

             skip_bits(gb, 2); //ref_select_code

         }

     }

     /* detect buggy encoders which don't set the low_delay flag (divx4/xvid/opendivx)*/

     // note we cannot detect divx5 without b-frames easily (although it's buggy too)

     if(s->vo_type==0 && s->vol_control_parameters==0 && s->divx_version==-1 && s->picture_number==0){

         av_log(s->avctx, AV_LOG_WARNING, "looks like this file was encoded with (divx4/(old)xvid/opendivx) -> forcing low_delay flag\n");

         s->low_delay=1;

     }



     s->picture_number++; // better than pic number==0 always ;)



     s->y_dc_scale_table= ff_mpeg4_y_dc_scale_table; //FIXME add short header support

     s->c_dc_scale_table= ff_mpeg4_c_dc_scale_table;



     if(s->workaround_bugs&FF_BUG_EDGE){

         s->h_edge_pos= s->width;

         s->v_edge_pos= s->height;

     }

     return 0;

}
