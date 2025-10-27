static int h261_decode_gob(H261Context *h){

    MpegEncContext * const s = &h->s;

    int v;

    

    ff_set_qscale(s, s->qscale);



    /* check for empty gob */

    v= show_bits(&s->gb, 15);



    if(get_bits_count(&s->gb) + 15 > s->gb.size_in_bits){

        v>>= get_bits_count(&s->gb) + 15 - s->gb.size_in_bits;

    }



    if(v==0){

        h261_decode_mb_skipped(h, 0, 33);

        return 0;

    }



    /* decode mb's */

    while(h->current_mba <= MAX_MBA)

    {

        int ret;

        /* DCT & quantize */

        ret= h261_decode_mb(h, s->block);

        if(ret<0){

            const int xy= s->mb_x + s->mb_y*s->mb_stride;

            if(ret==SLICE_END){

                MPV_decode_mb(s, s->block);

                if(h->loop_filter){

                    ff_h261_loop_filter(h);

                }

                h->loop_filter = 0;

                h261_decode_mb_skipped(h, h->current_mba-h->mba_diff, h->current_mba-1);

                h261_decode_mb_skipped(h, h->current_mba, 33);                

                return 0;

            }else if(ret==SLICE_NOEND){

                av_log(s->avctx, AV_LOG_ERROR, "Slice mismatch at MB: %d\n", xy);

                return -1;

            }

            av_log(s->avctx, AV_LOG_ERROR, "Error at MB: %d\n", xy);

            return -1;

        }

        MPV_decode_mb(s, s->block);

        if(h->loop_filter){

            ff_h261_loop_filter(h);

        }



        h->loop_filter = 0;

        h261_decode_mb_skipped(h, h->current_mba-h->mba_diff, h->current_mba-1);

    }

    

    return -1;

}
