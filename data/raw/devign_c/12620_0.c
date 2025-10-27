static inline int mpeg4_decode_dc(MpegEncContext * s, int n, int *dir_ptr)

{

    int level, pred, code;

    uint16_t *dc_val;



    if (n < 4) 

        code = get_vlc2(&s->gb, dc_lum.table, DC_VLC_BITS, 1);

    else 

        code = get_vlc2(&s->gb, dc_chrom.table, DC_VLC_BITS, 1);

    if (code < 0 || code > 9 /* && s->nbit<9 */){

        fprintf(stderr, "illegal dc vlc\n");

        return -1;

    }

    if (code == 0) {

        level = 0;

    } else {

        level = get_xbits(&s->gb, code);

        if (code > 8){

            if(get_bits1(&s->gb)==0){ /* marker */

                if(s->error_resilience>=2){

                    fprintf(stderr, "dc marker bit missing\n");

                    return -1;

                }

            }

        }

    }

    pred = ff_mpeg4_pred_dc(s, n, &dc_val, dir_ptr);

    level += pred;

    if (level < 0){

        if(s->error_resilience>=3){

            fprintf(stderr, "dc<0 at %dx%d\n", s->mb_x, s->mb_y);

            return -1;

        }

        level = 0;

    }

    if (n < 4) {

        *dc_val = level * s->y_dc_scale;

    } else {

        *dc_val = level * s->c_dc_scale;

    }

    if(s->error_resilience>=3){

        if(*dc_val > 2048 + s->y_dc_scale + s->c_dc_scale){

            fprintf(stderr, "dc overflow at %dx%d\n", s->mb_x, s->mb_y);

            return -1;

        }

    }

    return level;

}
