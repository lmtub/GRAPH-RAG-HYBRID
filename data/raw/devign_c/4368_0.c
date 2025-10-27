static int decode_main_header(NUTContext *nut){

    AVFormatContext *s= nut->avf;

    ByteIOContext *bc = &s->pb;

    uint64_t tmp;

    int i, j;

    

    get_packetheader(nut, bc, 8, 1);



    tmp = get_v(bc);

    if (tmp != 1){

	av_log(s, AV_LOG_ERROR, "bad version (%Ld)\n", tmp);

        return -1;

    }

    

    nut->stream_count = get_v(bc);

    get_v(bc); //checksum threshold



    for(i=0; i<256;){

        int tmp_flags = get_v(bc);

        int tmp_stream= get_v(bc);

        int tmp_mul   = get_v(bc);

        int tmp_size  = get_v(bc);

        int count     = get_v(bc);



        if(count == 0 || i+count > 256){

            av_log(s, AV_LOG_ERROR, "illegal count %d at %d\n", count, i);

            return -1;

        }



        if((tmp_flags & FLAG_FRAME_TYPE) && tmp_flags != 1){

            if(tmp_flags & FLAG_PRED_KEY_FRAME){

                av_log(s, AV_LOG_ERROR, "keyframe prediction in non 0 frame type\n");

                return -1;

            }

            if(!(tmp_flags & FLAG_PTS) || !(tmp_flags & FLAG_FULL_PTS) ){

                av_log(s, AV_LOG_ERROR, "no full pts in non 0 frame type\n");

                return -1;

            }

        }



        for(j=0; j<count; j++,i++){

            if(tmp_stream > nut->stream_count + 1){

                av_log(s, AV_LOG_ERROR, "illegal stream number\n");

                return -1;

            }



            nut->frame_code[i].flags           = tmp_flags ;

            nut->frame_code[i].stream_id_plus1 = tmp_stream;

            nut->frame_code[i].size_mul        = tmp_mul   ;

            nut->frame_code[i].size_lsb        = tmp_size  ;

            if(++tmp_size >= tmp_mul){

                tmp_size=0;

                tmp_stream++;

            }

        }

    }

    if(nut->frame_code['N'].flags != 1){

        av_log(s, AV_LOG_ERROR, "illegal frame_code table\n");

        return -1;

    }



    if(check_checksum(bc)){

        av_log(s, AV_LOG_ERROR, "Main header checksum missmatch\n");

        return -1;

    }



    return 0;

}
