static void vc1_decode_b_blocks(VC1Context *v)

{

    MpegEncContext *s = &v->s;



    /* select codingmode used for VLC tables selection */

    switch(v->c_ac_table_index){

    case 0:

        v->codingset = (v->pqindex <= 8) ? CS_HIGH_RATE_INTRA : CS_LOW_MOT_INTRA;

        break;

    case 1:

        v->codingset = CS_HIGH_MOT_INTRA;

        break;

    case 2:

        v->codingset = CS_MID_RATE_INTRA;

        break;

    }



    switch(v->c_ac_table_index){

    case 0:

        v->codingset2 = (v->pqindex <= 8) ? CS_HIGH_RATE_INTER : CS_LOW_MOT_INTER;

        break;

    case 1:

        v->codingset2 = CS_HIGH_MOT_INTER;

        break;

    case 2:

        v->codingset2 = CS_MID_RATE_INTER;

        break;

    }



    s->first_slice_line = 1;

    for(s->mb_y = 0; s->mb_y < s->mb_height; s->mb_y++) {

        for(s->mb_x = 0; s->mb_x < s->mb_width; s->mb_x++) {

            ff_init_block_index(s);

            ff_update_block_index(s);

            s->dsp.clear_blocks(s->block[0]);



            vc1_decode_b_mb(v);

            if(get_bits_count(&s->gb) > v->bits || get_bits_count(&s->gb) < 0) {

                ff_er_add_slice(s, 0, 0, s->mb_x, s->mb_y, (AC_END|DC_END|MV_END));

                av_log(s->avctx, AV_LOG_ERROR, "Bits overconsumption: %i > %i at %ix%i\n", get_bits_count(&s->gb), v->bits,s->mb_x,s->mb_y);

                return;

            }

            if(v->s.loop_filter) vc1_loop_filter_iblk(s, s->current_picture.qscale_table[s->mb_x + s->mb_y *s->mb_stride]);

        }

        ff_draw_horiz_band(s, s->mb_y * 16, 16);

        s->first_slice_line = 0;

    }

    ff_er_add_slice(s, 0, 0, s->mb_width - 1, s->mb_height - 1, (AC_END|DC_END|MV_END));

}
