static int nut_read_packet(AVFormatContext *s, AVPacket *pkt)

{

    NUTContext *nut = s->priv_data;

    ByteIOContext *bc = &s->pb;

    int i, frame_code=0, ret, skip;

    int64_t ts, back_ptr;



    for(;;){

        int64_t pos= url_ftell(bc);

        uint64_t tmp= nut->next_startcode;

        nut->next_startcode=0;



        if (url_feof(bc))

            return -1;



        if(tmp){

            pos-=8;

        }else{

            frame_code = get_byte(bc);

            if(frame_code == 'N'){

                tmp= frame_code;

                for(i=1; i<8; i++)

                    tmp = (tmp<<8) + get_byte(bc);

            }

        }

        switch(tmp){

        case MAIN_STARTCODE:

        case STREAM_STARTCODE:

        case INDEX_STARTCODE:

            skip= get_packetheader(nut, bc, 0);

            url_fseek(bc, skip, SEEK_CUR);

            break;

        case INFO_STARTCODE:

            if(decode_info_header(nut)<0)

                goto resync;

            break;

        case SYNCPOINT_STARTCODE:

            if(decode_syncpoint(nut, &ts, &back_ptr)<0)

                goto resync;

            frame_code = get_byte(bc);

        case 0:

            ret= decode_frame(nut, pkt, frame_code);

            if(ret==0)

                return 0;

            else if(ret==1) //ok but discard packet

                break;

        default:

resync:

av_log(s, AV_LOG_DEBUG, "syncing from %"PRId64"\n", pos);

            tmp= find_any_startcode(bc, nut->last_syncpoint_pos+1);

            if(tmp==0)

                return -1;

av_log(s, AV_LOG_DEBUG, "sync\n");

            nut->next_startcode= tmp;

        }

    }

}
