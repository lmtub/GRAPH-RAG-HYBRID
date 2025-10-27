static int mpeg1_find_frame_end(MpegEncContext *s, uint8_t *buf, int buf_size){

    ParseContext *pc= &s->parse_context;

    int i;

    uint32_t state;

    

    state= pc->state;

    

    i=0;

    if(!pc->frame_start_found){

        for(i=0; i<buf_size; i++){

            state= (state<<8) | buf[i];

            if(state >= SLICE_MIN_START_CODE && state <= SLICE_MAX_START_CODE){

                i++;

                pc->frame_start_found=1;

                break;

            }

        }

    }

    

    if(pc->frame_start_found){

        for(; i<buf_size; i++){

            state= (state<<8) | buf[i];

            if((state&0xFFFFFF00) == 0x100){

                if(state < SLICE_MIN_START_CODE || state > SLICE_MAX_START_CODE){

                    pc->frame_start_found=0;

                    pc->state=-1; 

                    return i-3;

                }

            }

        }

    }        

    pc->state= state;

    return -1;

}
