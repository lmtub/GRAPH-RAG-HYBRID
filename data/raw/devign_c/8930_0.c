static int ljpeg_decode_rgb_scan(MJpegDecodeContext *s, int predictor, int point_transform){

    int i, mb_x, mb_y;

    uint16_t (*buffer)[4];

    int left[3], top[3], topleft[3];

    const int linesize= s->linesize[0];

    const int mask= (1<<s->bits)-1;



    av_fast_malloc(&s->ljpeg_buffer, &s->ljpeg_buffer_size, (unsigned)s->mb_width * 4 * sizeof(s->ljpeg_buffer[0][0]));

    buffer= s->ljpeg_buffer;



    for(i=0; i<3; i++){

        buffer[0][i]= 1 << (s->bits + point_transform - 1);

    }

    for(mb_y = 0; mb_y < s->mb_height; mb_y++) {

        const int modified_predictor= mb_y ? predictor : 1;

        uint8_t *ptr = s->picture.data[0] + (linesize * mb_y);



        if (s->interlaced && s->bottom_field)

            ptr += linesize >> 1;



        for(i=0; i<3; i++){

            top[i]= left[i]= topleft[i]= buffer[0][i];

        }

        for(mb_x = 0; mb_x < s->mb_width; mb_x++) {

            if (s->restart_interval && !s->restart_count)

                s->restart_count = s->restart_interval;



            for(i=0;i<3;i++) {

                int pred;



                topleft[i]= top[i];

                top[i]= buffer[mb_x][i];



                PREDICT(pred, topleft[i], top[i], left[i], modified_predictor);



                left[i]=

                buffer[mb_x][i]= mask & (pred + (mjpeg_decode_dc(s, s->dc_index[i]) << point_transform));

            }



            if (s->restart_interval && !--s->restart_count) {

                align_get_bits(&s->gb);

                skip_bits(&s->gb, 16); /* skip RSTn */

            }

        }



        if(s->rct){

            for(mb_x = 0; mb_x < s->mb_width; mb_x++) {

                ptr[3*mb_x+1] = buffer[mb_x][0] - ((buffer[mb_x][1] + buffer[mb_x][2] - 0x200)>>2);

                ptr[3*mb_x+0] = buffer[mb_x][1] + ptr[3*mb_x+1];

                ptr[3*mb_x+2] = buffer[mb_x][2] + ptr[3*mb_x+1];

            }

        }else if(s->pegasus_rct){

            for(mb_x = 0; mb_x < s->mb_width; mb_x++) {

                ptr[3*mb_x+1] = buffer[mb_x][0] - ((buffer[mb_x][1] + buffer[mb_x][2])>>2);

                ptr[3*mb_x+0] = buffer[mb_x][1] + ptr[3*mb_x+1];

                ptr[3*mb_x+2] = buffer[mb_x][2] + ptr[3*mb_x+1];

            }

        }else{

            for(mb_x = 0; mb_x < s->mb_width; mb_x++) {

                ptr[3*mb_x+0] = buffer[mb_x][2];

                ptr[3*mb_x+1] = buffer[mb_x][1];

                ptr[3*mb_x+2] = buffer[mb_x][0];

            }

        }

    }

    return 0;

}
