void mpeg1_encode_picture_header(MpegEncContext *s, int picture_number)

{

    mpeg1_encode_sequence_header(s);



    /* mpeg1 picture header */

    put_header(s, PICTURE_START_CODE);

    /* temporal reference */



    // RAL: s->picture_number instead of s->fake_picture_number

    put_bits(&s->pb, 10, (s->picture_number - 

                          s->gop_picture_number) & 0x3ff); 

    s->fake_picture_number++;

    

    put_bits(&s->pb, 3, s->pict_type);



    s->vbv_delay_ptr= s->pb.buf + get_bit_count(&s->pb)/8;

    put_bits(&s->pb, 16, 0xFFFF); /* vbv_delay */

    

    // RAL: Forward f_code also needed for B frames

    if (s->pict_type == P_TYPE || s->pict_type == B_TYPE) {

        put_bits(&s->pb, 1, 0); /* half pel coordinates */

        if(s->codec_id == CODEC_ID_MPEG1VIDEO)

            put_bits(&s->pb, 3, s->f_code); /* forward_f_code */

        else

            put_bits(&s->pb, 3, 7); /* forward_f_code */

    }

    

    // RAL: Backward f_code necessary for B frames

    if (s->pict_type == B_TYPE) {

        put_bits(&s->pb, 1, 0); /* half pel coordinates */

        if(s->codec_id == CODEC_ID_MPEG1VIDEO)

            put_bits(&s->pb, 3, s->b_code); /* backward_f_code */

        else

            put_bits(&s->pb, 3, 7); /* backward_f_code */

    }



    put_bits(&s->pb, 1, 0); /* extra bit picture */



    s->frame_pred_frame_dct = 1;

    if(s->codec_id == CODEC_ID_MPEG2VIDEO){

        put_header(s, EXT_START_CODE);

        put_bits(&s->pb, 4, 8); //pic ext

        if (s->pict_type == P_TYPE || s->pict_type == B_TYPE) {

            put_bits(&s->pb, 4, s->f_code);

            put_bits(&s->pb, 4, s->f_code);

        }else{

            put_bits(&s->pb, 8, 255);

        }

        if (s->pict_type == B_TYPE) {

            put_bits(&s->pb, 4, s->b_code);

            put_bits(&s->pb, 4, s->b_code);

        }else{

            put_bits(&s->pb, 8, 255);

        }

        put_bits(&s->pb, 2, s->intra_dc_precision);

        put_bits(&s->pb, 2, s->picture_structure= PICT_FRAME);

        if (s->progressive_sequence) {

            put_bits(&s->pb, 1, 0); /* no repeat */

        } else {

            put_bits(&s->pb, 1, s->current_picture_ptr->top_field_first);

        }

        /* XXX: optimize the generation of this flag with entropy

           measures */

        s->frame_pred_frame_dct = s->progressive_sequence;

        

        put_bits(&s->pb, 1, s->frame_pred_frame_dct);

        put_bits(&s->pb, 1, s->concealment_motion_vectors);

        put_bits(&s->pb, 1, s->q_scale_type);

        put_bits(&s->pb, 1, s->intra_vlc_format);

        put_bits(&s->pb, 1, s->alternate_scan);

        put_bits(&s->pb, 1, s->repeat_first_field);

        put_bits(&s->pb, 1, s->chroma_420_type=1);

        s->progressive_frame = s->progressive_sequence;

        put_bits(&s->pb, 1, s->progressive_frame);

        put_bits(&s->pb, 1, 0); //composite_display_flag

    }

    if(s->flags & CODEC_FLAG_SVCD_SCAN_OFFSET){

        int i;



        put_header(s, USER_START_CODE);

        for(i=0; i<sizeof(svcd_scan_offset_placeholder); i++){

            put_bits(&s->pb, 8, svcd_scan_offset_placeholder[i]);

        }

    }

    

    s->mb_y=0;

    ff_mpeg1_encode_slice_header(s);

}
