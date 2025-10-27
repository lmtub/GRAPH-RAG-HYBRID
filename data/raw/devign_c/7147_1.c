static void set_frame_distances(MpegEncContext * s){

    assert(s->current_picture_ptr->f.pts != AV_NOPTS_VALUE);

    s->time = s->current_picture_ptr->f.pts * s->avctx->time_base.num;



    if(s->pict_type==AV_PICTURE_TYPE_B){

        s->pb_time= s->pp_time - (s->last_non_b_time - s->time);

        assert(s->pb_time > 0 && s->pb_time < s->pp_time);

    }else{

        s->pp_time= s->time - s->last_non_b_time;

        s->last_non_b_time= s->time;

        assert(s->picture_number==0 || s->pp_time > 0);

    }

}
