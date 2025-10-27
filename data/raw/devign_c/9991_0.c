static int decode_group3_2d_line(AVCodecContext *avctx, GetBitContext *gb,

                                 int width, int *runs, const int *runend, const int *ref)

{

    int mode = 0, offs = 0, run = 0, saved_run = 0, t;

    int run_off = *ref++;

    int *run_start = runs;



    runend--; // for the last written 0



    while(offs < width){

        int cmode = get_vlc2(gb, ccitt_group3_2d_vlc.table, 9, 1);

        if(cmode == -1){

            av_log(avctx, AV_LOG_ERROR, "Incorrect mode VLC\n");

            return -1;

        }

        //sync line pointers

        if(runs != run_start)while(run_off <= offs){

            run_off += *ref++;

            run_off += *ref++;

        }

        if(!cmode){//pass mode

            run_off += *ref++;

            run = run_off - offs;

            run_off += *ref++;

            offs += run;

            if(offs > width){

                av_log(avctx, AV_LOG_ERROR, "Run went out of bounds\n");

                return -1;

            }

            saved_run += run;

        }else if(cmode == 1){//horizontal mode

            int k;

            for(k = 0; k < 2; k++){

                run = 0;

                for(;;){

                    t = get_vlc2(gb, ccitt_vlc[mode].table, 9, 2);

                    if(t == -1){

                        av_log(avctx, AV_LOG_ERROR, "Incorrect code\n");

                        return -1;

                    }

                    run += t;

                    if(t < 64)

                        break;

                }

                *runs++ = run + saved_run;

                if(runs >= runend){

                    av_log(avctx, AV_LOG_ERROR, "Run overrun\n");

                    return -1;

                }

                saved_run = 0;

                offs += run;

                if(offs > width){

                    av_log(avctx, AV_LOG_ERROR, "Run went out of bounds\n");

                    return -1;

                }

                mode = !mode;

            }

        }else if(cmode == 9 || cmode == 10){

            av_log(avctx, AV_LOG_ERROR, "Special modes are not supported (yet)\n");

            return -1;

        }else{//vertical mode

            run = run_off - offs + (cmode - 5);

            if(cmode >= 5)

                run_off += *ref++;

            else

                run_off -= *--ref;

            offs += run;

            if(offs > width){

                av_log(avctx, AV_LOG_ERROR, "Run went out of bounds\n");

                return -1;

            }

            *runs++ = run + saved_run;

            if(runs >= runend){

                av_log(avctx, AV_LOG_ERROR, "Run overrun\n");

                return -1;

            }

            saved_run = 0;

            mode = !mode;

        }

    }

    *runs++ = saved_run;

    *runs++ = 0;

    return 0;

}
