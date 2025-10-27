static int decode_syncpoint(NUTContext *nut){

    AVFormatContext *s= nut->avf;

    ByteIOContext *bc = &s->pb;

    int64_t end;

    uint64_t tmp;

    int i;

    AVRational time_base;



    nut->last_syncpoint_pos= url_ftell(bc)-8;



    end= get_packetheader(nut, bc, 1);

    end += url_ftell(bc) - 4;



    tmp= get_v(bc);

    get_v(bc); //back_ptr_div16



    time_base= nut->time_base[tmp % nut->time_base_count];

    for(i=0; i<s->nb_streams; i++){

        nut->stream[i].last_pts= av_rescale_rnd(

            tmp / nut->time_base_count,

            time_base.num * (int64_t)nut->stream[i].time_base.den,

            time_base.den * (int64_t)nut->stream[i].time_base.num,

            AV_ROUND_DOWN);

        //last_key_frame ?

    }

    //FIXME put this in a reset func maybe



    if(skip_reserved(bc, end) || check_checksum(bc)){

        av_log(s, AV_LOG_ERROR, "sync point checksum mismatch\n");

        return -1;

    }

    return 0;

}
