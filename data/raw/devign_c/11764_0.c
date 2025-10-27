static int rv10_decode_packet(AVCodecContext *avctx, 

                             UINT8 *buf, int buf_size)

{

    MpegEncContext *s = avctx->priv_data;

    int i, mb_count, mb_pos, left;



    init_get_bits(&s->gb, buf, buf_size);

    

    mb_count = rv10_decode_picture_header(s);

    if (mb_count < 0) {

        fprintf(stderr, "HEADER ERROR\n");

        return -1;

    }

    

    if (s->mb_x >= s->mb_width ||

        s->mb_y >= s->mb_height) {

        fprintf(stderr, "POS ERROR %d %d\n", s->mb_x, s->mb_y);

        return -1;

    }

    mb_pos = s->mb_y * s->mb_width + s->mb_x;

    left = s->mb_width * s->mb_height - mb_pos;

    if (mb_count > left) {

        fprintf(stderr, "COUNT ERROR\n");

        return -1;

    }



    if (s->mb_x == 0 && s->mb_y == 0) {

        if(MPV_frame_start(s, avctx) < 0)

            return -1;

    }



#ifdef DEBUG

    printf("qscale=%d\n", s->qscale);

#endif



    /* default quantization values */

    s->y_dc_scale = 8;

    s->c_dc_scale = 8;

    s->rv10_first_dc_coded[0] = 0;

    s->rv10_first_dc_coded[1] = 0;

    s->rv10_first_dc_coded[2] = 0;



    if(s->mb_y==0) s->first_slice_line=1;

    

    s->block_wrap[0]=

    s->block_wrap[1]=

    s->block_wrap[2]=

    s->block_wrap[3]= s->mb_width*2 + 2;

    s->block_wrap[4]=

    s->block_wrap[5]= s->mb_width + 2;

    ff_init_block_index(s);

    /* decode each macroblock */

    for(i=0;i<mb_count;i++) {

        ff_update_block_index(s);

#ifdef DEBUG

        printf("**mb x=%d y=%d\n", s->mb_x, s->mb_y);

#endif

        

	s->dsp.clear_blocks(s->block[0]);

        s->mv_dir = MV_DIR_FORWARD;

        s->mv_type = MV_TYPE_16X16; 

        if (ff_h263_decode_mb(s, s->block) == SLICE_ERROR) {

            fprintf(stderr, "ERROR at MB %d %d\n", s->mb_x, s->mb_y);

            return -1;

        }

        MPV_decode_mb(s, s->block);

        if (++s->mb_x == s->mb_width) {

            s->mb_x = 0;

            s->mb_y++;

            ff_init_block_index(s);

            s->first_slice_line=0;

        }

    }



    return buf_size;

}
