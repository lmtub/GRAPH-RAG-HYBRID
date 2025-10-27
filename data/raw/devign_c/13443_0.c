int ff_h263_resync(MpegEncContext *s){

    int left, ret;

    

    if(s->codec_id==CODEC_ID_MPEG4)

        skip_bits1(&s->gb);

    

    align_get_bits(&s->gb);



    if(show_bits(&s->gb, 16)==0){

        if(s->codec_id==CODEC_ID_MPEG4)

            ret= mpeg4_decode_video_packet_header(s);

        else

            ret= h263_decode_gob_header(s);

        if(ret>=0)

            return 0;

    }

    //ok, its not where its supposed to be ...

    s->gb= s->last_resync_gb;

    align_get_bits(&s->gb);

    left= s->gb.size*8 - get_bits_count(&s->gb);

    

    for(;left>16+1+5+5; left-=8){ 

        if(show_bits(&s->gb, 16)==0){

            GetBitContext bak= s->gb;



            if(s->codec_id==CODEC_ID_MPEG4)

                ret= mpeg4_decode_video_packet_header(s);

            else

                ret= h263_decode_gob_header(s);

            if(ret>=0)

                return 0;



            s->gb= bak;

        }

        skip_bits(&s->gb, 8);

    }

    

    return -1;

}
