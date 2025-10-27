int ff_wmv2_decode_secondary_picture_header(MpegEncContext *s)

{

    Wmv2Context *const w = (Wmv2Context *) s;



    if (s->pict_type == AV_PICTURE_TYPE_I) {

        if (w->j_type_bit)

            w->j_type = get_bits1(&s->gb);

        else

            w->j_type = 0; // FIXME check



        if (!w->j_type) {

            if (w->per_mb_rl_bit)

                s->per_mb_rl_table = get_bits1(&s->gb);

            else

                s->per_mb_rl_table = 0;



            if (!s->per_mb_rl_table) {

                s->rl_chroma_table_index = decode012(&s->gb);

                s->rl_table_index        = decode012(&s->gb);

            }



            s->dc_table_index = get_bits1(&s->gb);

        }

        s->inter_intra_pred = 0;

        s->no_rounding      = 1;

        if (s->avctx->debug & FF_DEBUG_PICT_INFO) {

            av_log(s->avctx, AV_LOG_DEBUG,

                   "qscale:%d rlc:%d rl:%d dc:%d mbrl:%d j_type:%d \n",

                   s->qscale, s->rl_chroma_table_index, s->rl_table_index,

                   s->dc_table_index, s->per_mb_rl_table, w->j_type);

        }

    } else {

        int cbp_index;

        w->j_type = 0;



        parse_mb_skip(w);

        cbp_index = decode012(&s->gb);

        w->cbp_table_index = wmv2_get_cbp_table_index(s, cbp_index);



        if (w->mspel_bit)

            s->mspel = get_bits1(&s->gb);

        else

            s->mspel = 0; // FIXME check



        if (w->abt_flag) {

            w->per_mb_abt = get_bits1(&s->gb) ^ 1;

            if (!w->per_mb_abt)

                w->abt_type = decode012(&s->gb);

        }



        if (w->per_mb_rl_bit)

            s->per_mb_rl_table = get_bits1(&s->gb);

        else

            s->per_mb_rl_table = 0;



        if (!s->per_mb_rl_table) {

            s->rl_table_index        = decode012(&s->gb);

            s->rl_chroma_table_index = s->rl_table_index;

        }



        s->dc_table_index   = get_bits1(&s->gb);

        s->mv_table_index   = get_bits1(&s->gb);



        s->inter_intra_pred = 0; // (s->width * s->height < 320 * 240 && s->bit_rate <= II_BITRATE);

        s->no_rounding     ^= 1;



        if (s->avctx->debug & FF_DEBUG_PICT_INFO) {

            av_log(s->avctx, AV_LOG_DEBUG,

                   "rl:%d rlc:%d dc:%d mv:%d mbrl:%d qp:%d mspel:%d "

                   "per_mb_abt:%d abt_type:%d cbp:%d ii:%d\n",

                   s->rl_table_index, s->rl_chroma_table_index,

                   s->dc_table_index, s->mv_table_index,

                   s->per_mb_rl_table, s->qscale, s->mspel,

                   w->per_mb_abt, w->abt_type, w->cbp_table_index,

                   s->inter_intra_pred);

        }

    }

    s->esc3_level_length = 0;

    s->esc3_run_length   = 0;

    s->picture_number++; // FIXME ?



    if (w->j_type) {

        ff_intrax8_decode_picture(&w->x8, &s->current_picture,

                                  &s->gb, &s->mb_x, &s->mb_y,

                                  2 * s->qscale, (s->qscale - 1) | 1,

                                  s->loop_filter, s->low_delay);



        ff_er_add_slice(&w->s.er, 0, 0,

                        (w->s.mb_x >> 1) - 1, (w->s.mb_y >> 1) - 1,

                        ER_MB_END);

        return 1;

    }



    return 0;

}
