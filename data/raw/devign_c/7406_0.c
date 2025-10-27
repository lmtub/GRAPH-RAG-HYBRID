static void mpeg4_encode_visual_object_header(MpegEncContext * s){

    int profile_and_level_indication;

    int vo_ver_id;

    

    if(s->max_b_frames || s->quarter_sample){

        profile_and_level_indication= 0xF1; // adv simple level 1

        vo_ver_id= 5;

    }else{

        profile_and_level_indication= 0x01; // simple level 1

        vo_ver_id= 1;

    }

    //FIXME levels



    put_bits(&s->pb, 16, 0);

    put_bits(&s->pb, 16, VOS_STARTCODE);



    put_bits(&s->pb, 8, profile_and_level_indication);



    put_bits(&s->pb, 16, 0);

    put_bits(&s->pb, 16, VISUAL_OBJ_STARTCODE);

    

    put_bits(&s->pb, 1, 1);

        put_bits(&s->pb, 4, vo_ver_id);

        put_bits(&s->pb, 3, 1); //priority

 

    put_bits(&s->pb, 4, 1); //visual obj type== video obj

    

    put_bits(&s->pb, 1, 0); //video signal type == no clue //FIXME



    ff_mpeg4_stuffing(&s->pb);

}
