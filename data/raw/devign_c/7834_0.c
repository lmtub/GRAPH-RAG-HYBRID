int ff_mpeg1_find_frame_end(ParseContext *pc, const uint8_t *buf, int buf_size)

{

    int i;

    uint32_t state= pc->state;



    /* EOF considered as end of frame */

    if (buf_size == 0)

        return 0;



/*

 0  frame start         -> 1/4

 1  first_SEQEXT        -> 0/2

 2  first field start   -> 3/0

 3  second_SEQEXT       -> 2/0

 4  searching end

*/



    for(i=0; i<buf_size; i++){

        assert(pc->frame_start_found>=0 && pc->frame_start_found<=4);

        if(pc->frame_start_found&1){

            if(state == EXT_START_CODE && (buf[i]&0xF0) != 0x80)

                pc->frame_start_found--;

            else if(state == EXT_START_CODE+2){

                if((buf[i]&3) == 3) pc->frame_start_found= 0;

                else                pc->frame_start_found= (pc->frame_start_found+1)&3;

            }

            state++;

        }else{

            i= ff_find_start_code(buf+i, buf+buf_size, &state) - buf - 1;

            if(pc->frame_start_found==0 && state >= SLICE_MIN_START_CODE && state <= SLICE_MAX_START_CODE){

                i++;

                pc->frame_start_found=4;

            }

            if(state == SEQ_END_CODE){

                pc->state=-1;

                return i+1;

            }

            if(pc->frame_start_found==2 && state == SEQ_START_CODE)

                pc->frame_start_found= 0;

            if(pc->frame_start_found<4 && state == EXT_START_CODE)

                pc->frame_start_found++;

            if(pc->frame_start_found == 4 && (state&0xFFFFFF00) == 0x100){

                if(state < SLICE_MIN_START_CODE || state > SLICE_MAX_START_CODE){

                    pc->frame_start_found=0;

                    pc->state=-1;

                    return i-3;

                }

            }

        }

    }

    pc->state= state;

    return END_NOT_FOUND;

}
