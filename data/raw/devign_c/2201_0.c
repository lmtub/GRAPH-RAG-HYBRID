static inline int msmpeg4_decode_block(MpegEncContext * s, DCTELEM * block,

                              int n, int coded, const uint8_t *scan_table)

{

    int level, i, last, run, run_diff;

    int dc_pred_dir;

    RLTable *rl;

    RL_VLC_ELEM *rl_vlc;

    int qmul, qadd;



    if (s->mb_intra) {

        qmul=1;

        qadd=0;



	/* DC coef */

        set_stat(ST_DC);

        level = msmpeg4_decode_dc(s, n, &dc_pred_dir);

#ifdef PRINT_MB

{

    static int c;

    if(n==0) c=0;

    if(n==4) printf("%X", c);

    c+= c +dc_pred_dir;

}

#endif

        if (level < 0){

            fprintf(stderr, "dc overflow- block: %d qscale: %d//\n", n, s->qscale);

            if(s->inter_intra_pred) level=0;

            else                    return -1;

        }

        if (n < 4) {

            rl = &rl_table[s->rl_table_index];

            if(level > 256*s->y_dc_scale){

                fprintf(stderr, "dc overflow+ L qscale: %d//\n", s->qscale);

                if(!s->inter_intra_pred) return -1;

            }

        } else {

            rl = &rl_table[3 + s->rl_chroma_table_index];

            if(level > 256*s->c_dc_scale){

                fprintf(stderr, "dc overflow+ C qscale: %d//\n", s->qscale);

                if(!s->inter_intra_pred) return -1;

            }

        }

        block[0] = level;



        run_diff = 0;

        i = 0;

        if (!coded) {

            goto not_coded;

        }

        if (s->ac_pred) {

            if (dc_pred_dir == 0) 

                scan_table = s->intra_v_scantable.permutated; /* left */

            else

                scan_table = s->intra_h_scantable.permutated; /* top */

        } else {

            scan_table = s->intra_scantable.permutated;

        }

        set_stat(ST_INTRA_AC);

        rl_vlc= rl->rl_vlc[0];

    } else {

        qmul = s->qscale << 1;

        qadd = (s->qscale - 1) | 1;

        i = -1;

        rl = &rl_table[3 + s->rl_table_index];



        if(s->msmpeg4_version==2)

            run_diff = 0;

        else

            run_diff = 1;



        if (!coded) {

            s->block_last_index[n] = i;

            return 0;

        }

        if(!scan_table)

            scan_table = s->inter_scantable.permutated;

        set_stat(ST_INTER_AC);

        rl_vlc= rl->rl_vlc[s->qscale];

    }

  {

    OPEN_READER(re, &s->gb);

    for(;;) {

        UPDATE_CACHE(re, &s->gb);

        GET_RL_VLC(level, run, re, &s->gb, rl_vlc, TEX_VLC_BITS, 2);

        if (level==0) {

            int cache;

            cache= GET_CACHE(re, &s->gb);

            /* escape */

            if (s->msmpeg4_version==1 || (cache&0x80000000)==0) {

                if (s->msmpeg4_version==1 || (cache&0x40000000)==0) {

                    /* third escape */

                    if(s->msmpeg4_version!=1) LAST_SKIP_BITS(re, &s->gb, 2);

                    UPDATE_CACHE(re, &s->gb);

                    if(s->msmpeg4_version<=3){

                        last=  SHOW_UBITS(re, &s->gb, 1); SKIP_CACHE(re, &s->gb, 1);

                        run=   SHOW_UBITS(re, &s->gb, 6); SKIP_CACHE(re, &s->gb, 6);

                        level= SHOW_SBITS(re, &s->gb, 8); LAST_SKIP_CACHE(re, &s->gb, 8);

                        SKIP_COUNTER(re, &s->gb, 1+6+8);

                    }else{                        

                        int sign;

                        last=  SHOW_UBITS(re, &s->gb, 1); SKIP_BITS(re, &s->gb, 1);

                        if(!s->esc3_level_length){

                            int ll;

                            //printf("ESC-3 %X at %d %d\n", show_bits(&s->gb, 24), s->mb_x, s->mb_y);

                            if(s->qscale<8){

                                ll= SHOW_UBITS(re, &s->gb, 3); SKIP_BITS(re, &s->gb, 3);

                                if(ll==0){

                                    if(SHOW_UBITS(re, &s->gb, 1)) printf("cool a new vlc code ,contact the ffmpeg developers and upload the file\n");

                                    SKIP_BITS(re, &s->gb, 1);

                                    ll=8;

                                }

                            }else{

                                ll=2;

                                while(ll<8 && SHOW_UBITS(re, &s->gb, 1)==0){

                                    ll++;

                                    SKIP_BITS(re, &s->gb, 1);

                                }

                                if(ll<8) SKIP_BITS(re, &s->gb, 1);

                            }



                            s->esc3_level_length= ll;

                            s->esc3_run_length= SHOW_UBITS(re, &s->gb, 2) + 3; SKIP_BITS(re, &s->gb, 2);

//printf("level length:%d, run length: %d\n", ll, s->esc3_run_length);

                            UPDATE_CACHE(re, &s->gb);

                        }

                        run=   SHOW_UBITS(re, &s->gb, s->esc3_run_length); 

                        SKIP_BITS(re, &s->gb, s->esc3_run_length);

                        

                        sign=  SHOW_UBITS(re, &s->gb, 1); 

                        SKIP_BITS(re, &s->gb, 1);

                        

                        level= SHOW_UBITS(re, &s->gb, s->esc3_level_length); 

                        SKIP_BITS(re, &s->gb, s->esc3_level_length);

                        if(sign) level= -level;

                    }

//printf("level: %d, run: %d at %d %d\n", level, run, s->mb_x, s->mb_y);

#if 0 // waste of time / this will detect very few errors

                    {

                        const int abs_level= ABS(level);

                        const int run1= run - rl->max_run[last][abs_level] - run_diff;

                        if(abs_level<=MAX_LEVEL && run<=MAX_RUN){

                            if(abs_level <= rl->max_level[last][run]){

                                fprintf(stderr, "illegal 3. esc, vlc encoding possible\n");

                                return DECODING_AC_LOST;

                            }

                            if(abs_level <= rl->max_level[last][run]*2){

                                fprintf(stderr, "illegal 3. esc, esc 1 encoding possible\n");

                                return DECODING_AC_LOST;

                            }

                            if(run1>=0 && abs_level <= rl->max_level[last][run1]){

                                fprintf(stderr, "illegal 3. esc, esc 2 encoding possible\n");

                                return DECODING_AC_LOST;

                            }

                        }

                    }

#endif

		    //level = level * qmul + (level>0) * qadd - (level<=0) * qadd ;

		    if (level>0) level= level * qmul + qadd;

                    else         level= level * qmul - qadd;

#if 0 // waste of time too :(

                    if(level>2048 || level<-2048){

                        fprintf(stderr, "|level| overflow in 3. esc\n");

                        return DECODING_AC_LOST;

                    }

#endif

                    i+= run + 1;

                    if(last) i+=192;

#ifdef ERROR_DETAILS

                if(run==66)

                    fprintf(stderr, "illegal vlc code in ESC3 level=%d\n", level);

                else if((i>62 && i<192) || i>192+63)

                    fprintf(stderr, "run overflow in ESC3 i=%d run=%d level=%d\n", i, run, level);

#endif

                } else {

                    /* second escape */

#if MIN_CACHE_BITS < 23

                    LAST_SKIP_BITS(re, &s->gb, 2);

                    UPDATE_CACHE(re, &s->gb);

#else

                    SKIP_BITS(re, &s->gb, 2);

#endif

                    GET_RL_VLC(level, run, re, &s->gb, rl_vlc, TEX_VLC_BITS, 2);

                    i+= run + rl->max_run[run>>7][level/qmul] + run_diff; //FIXME opt indexing

                    level = (level ^ SHOW_SBITS(re, &s->gb, 1)) - SHOW_SBITS(re, &s->gb, 1);

                    LAST_SKIP_BITS(re, &s->gb, 1);

#ifdef ERROR_DETAILS

                if(run==66)

                    fprintf(stderr, "illegal vlc code in ESC2 level=%d\n", level);

                else if((i>62 && i<192) || i>192+63)

                    fprintf(stderr, "run overflow in ESC2 i=%d run=%d level=%d\n", i, run, level);

#endif

                }

            } else {

                /* first escape */

#if MIN_CACHE_BITS < 22

                LAST_SKIP_BITS(re, &s->gb, 1);

                UPDATE_CACHE(re, &s->gb);

#else

                SKIP_BITS(re, &s->gb, 1);

#endif

                GET_RL_VLC(level, run, re, &s->gb, rl_vlc, TEX_VLC_BITS, 2);

                i+= run;

                level = level + rl->max_level[run>>7][(run-1)&63] * qmul;//FIXME opt indexing

                level = (level ^ SHOW_SBITS(re, &s->gb, 1)) - SHOW_SBITS(re, &s->gb, 1);

                LAST_SKIP_BITS(re, &s->gb, 1);

#ifdef ERROR_DETAILS

                if(run==66)

                    fprintf(stderr, "illegal vlc code in ESC1 level=%d\n", level);

                else if((i>62 && i<192) || i>192+63)

                    fprintf(stderr, "run overflow in ESC1 i=%d run=%d level=%d\n", i, run, level);

#endif

            }

        } else {

            i+= run;

            level = (level ^ SHOW_SBITS(re, &s->gb, 1)) - SHOW_SBITS(re, &s->gb, 1);

            LAST_SKIP_BITS(re, &s->gb, 1);

#ifdef ERROR_DETAILS

                if(run==66)

                    fprintf(stderr, "illegal vlc code level=%d\n", level);

                else if((i>62 && i<192) || i>192+63)

                    fprintf(stderr, "run overflow i=%d run=%d level=%d\n", i, run, level);

#endif

        }

        if (i > 62){

            i-= 192;

            if(i&(~63)){

                const int left= s->gb.size*8 - get_bits_count(&s->gb);

                if(((i+192 == 64 && level/qmul==-1) || s->error_resilience<=1) && left>=0){

                    fprintf(stderr, "ignoring overflow at %d %d\n", s->mb_x, s->mb_y);

                    break;

                }else{

                    fprintf(stderr, "ac-tex damaged at %d %d\n", s->mb_x, s->mb_y);

                    return -1;

                }

            }



            block[scan_table[i]] = level;

            break;

        }



        block[scan_table[i]] = level;

    }

    CLOSE_READER(re, &s->gb);

  }

 not_coded:

    if (s->mb_intra) {

        mpeg4_pred_ac(s, block, n, dc_pred_dir);

        if (s->ac_pred) {

            i = 63; /* XXX: not optimal */

        }

    }

    if(s->msmpeg4_version>=4 && i>0) i=63; //FIXME/XXX optimize

    s->block_last_index[n] = i;

    

    return 0;

}
