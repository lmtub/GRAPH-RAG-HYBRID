static void vc1_decode_p_blocks(VC1Context *v)

{

    MpegEncContext *s = &v->s;

    int apply_loop_filter;



    /* select codingmode used for VLC tables selection */

    switch (v->c_ac_table_index) {

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



    switch (v->c_ac_table_index) {

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



    apply_loop_filter   = s->loop_filter && !(s->avctx->skip_loop_filter >= AVDISCARD_NONKEY);

    s->first_slice_line = 1;

    memset(v->cbp_base, 0, sizeof(v->cbp_base[0])*2*s->mb_stride);

    for (s->mb_y = s->start_mb_y; s->mb_y < s->end_mb_y; s->mb_y++) {

        s->mb_x = 0;

        ff_init_block_index(s);

        for (; s->mb_x < s->mb_width; s->mb_x++) {

            ff_update_block_index(s);



            if (v->fcm == ILACE_FIELD)

                vc1_decode_p_mb_intfi(v);

            else if (v->fcm == ILACE_FRAME)

                vc1_decode_p_mb_intfr(v);

            else vc1_decode_p_mb(v);

            if (s->mb_y != s->start_mb_y && apply_loop_filter && v->fcm == PROGRESSIVE)

                vc1_apply_p_loop_filter(v);

            if (get_bits_count(&s->gb) > v->bits || get_bits_count(&s->gb) < 0) {

                // TODO: may need modification to handle slice coding

                ff_er_add_slice(s, 0, s->start_mb_y, s->mb_x, s->mb_y, ER_MB_ERROR);

                av_log(s->avctx, AV_LOG_ERROR, "Bits overconsumption: %i > %i at %ix%i\n",

                       get_bits_count(&s->gb), v->bits, s->mb_x, s->mb_y);

                return;

            }

        }

        memmove(v->cbp_base,      v->cbp,      sizeof(v->cbp_base[0])      * s->mb_stride);

        memmove(v->ttblk_base,    v->ttblk,    sizeof(v->ttblk_base[0])    * s->mb_stride);

        memmove(v->is_intra_base, v->is_intra, sizeof(v->is_intra_base[0]) * s->mb_stride);

        memmove(v->luma_mv_base,  v->luma_mv,  sizeof(v->luma_mv_base[0])  * s->mb_stride);

        if (s->mb_y != s->start_mb_y) ff_draw_horiz_band(s, (s->mb_y - 1) * 16, 16);

        s->first_slice_line = 0;

    }

    if (apply_loop_filter) {

        s->mb_x = 0;

        ff_init_block_index(s);

        for (; s->mb_x < s->mb_width; s->mb_x++) {

            ff_update_block_index(s);

            vc1_apply_p_loop_filter(v);

        }

    }

    if (s->end_mb_y >= s->start_mb_y)

        ff_draw_horiz_band(s, (s->end_mb_y - 1) * 16, 16);

    ff_er_add_slice(s, 0, s->start_mb_y << v->field_mode, s->mb_width - 1,

                    (s->end_mb_y << v->field_mode) - 1, ER_MB_END);

}
