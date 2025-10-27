static int msmpeg4_decode_dc(MpegEncContext * s, int n, int *dir_ptr)

{

    int level, pred;



    if(s->msmpeg4_version<=2){

        if (n < 4) {

            level = get_vlc2(&s->gb, v2_dc_lum_vlc.table, DC_VLC_BITS, 3);

        } else {

            level = get_vlc2(&s->gb, v2_dc_chroma_vlc.table, DC_VLC_BITS, 3);

        }

        if (level < 0)

            return -1;

        level-=256;

    }else{  //FIXME optimize use unified tables & index

        if (n < 4) {

            level = get_vlc2(&s->gb, ff_msmp4_dc_luma_vlc[s->dc_table_index].table, DC_VLC_BITS, 3);

        } else {

            level = get_vlc2(&s->gb, ff_msmp4_dc_chroma_vlc[s->dc_table_index].table, DC_VLC_BITS, 3);

        }

        if (level < 0){

            av_log(s->avctx, AV_LOG_ERROR, "illegal dc vlc\n");

            return -1;

        }



        if (level == DC_MAX) {

            level = get_bits(&s->gb, 8);

            if (get_bits1(&s->gb))

                level = -level;

        } else if (level != 0) {

            if (get_bits1(&s->gb))

                level = -level;

        }

    }



    if(s->msmpeg4_version==1){

        int32_t *dc_val;

        pred = msmpeg4v1_pred_dc(s, n, &dc_val);

        level += pred;



        /* update predictor */

        *dc_val= level;

    }else{

        int16_t *dc_val;

        pred   = ff_msmpeg4_pred_dc(s, n, &dc_val, dir_ptr);

        level += pred;



        /* update predictor */

        if (n < 4) {

            *dc_val = level * s->y_dc_scale;

        } else {

            *dc_val = level * s->c_dc_scale;

        }

    }



    return level;

}
