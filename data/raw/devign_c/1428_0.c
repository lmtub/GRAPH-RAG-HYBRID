void ff_fix_long_p_mvs(MpegEncContext * s)

{

    const int f_code= s->f_code;

    int y;

    UINT8 * fcode_tab= s->fcode_tab;



    /* clip / convert to intra 16x16 type MVs */

    for(y=0; y<s->mb_height; y++){

        int x;

        int xy= (y+1)* (s->mb_width+2)+1;

        int i= y*s->mb_width;

        for(x=0; x<s->mb_width; x++){

            if(s->mb_type[i]&MB_TYPE_INTER){

                if(   fcode_tab[s->p_mv_table[xy][0] + MAX_MV] > f_code

                   || fcode_tab[s->p_mv_table[xy][0] + MAX_MV] == 0

                   || fcode_tab[s->p_mv_table[xy][1] + MAX_MV] > f_code

                   || fcode_tab[s->p_mv_table[xy][1] + MAX_MV] == 0 ){

                    s->mb_type[i] &= ~MB_TYPE_INTER;

                    s->mb_type[i] |= MB_TYPE_INTRA;

                    s->p_mv_table[xy][0] = 0;

                    s->p_mv_table[xy][1] = 0;

                }

            }

            xy++;

            i++;

        }

    }



    if(s->flags&CODEC_FLAG_4MV){

        const int wrap= 2+ s->mb_width*2;



        /* clip / convert to intra 8x8 type MVs */

        for(y=0; y<s->mb_height; y++){

            int xy= (y*2 + 1)*wrap + 1;

            int i= y*s->mb_width;

            int x;



            for(x=0; x<s->mb_width; x++){

                if(s->mb_type[i]&MB_TYPE_INTER4V){

                    int block;

                    for(block=0; block<4; block++){

                        int off= (block& 1) + (block>>1)*wrap;

                        int mx= s->motion_val[ xy + off ][0];

                        int my= s->motion_val[ xy + off ][1];



                        if(   fcode_tab[mx + MAX_MV] > f_code

                           || fcode_tab[mx + MAX_MV] == 0

                           || fcode_tab[my + MAX_MV] > f_code

                           || fcode_tab[my + MAX_MV] == 0 ){

                            s->mb_type[i] &= ~MB_TYPE_INTER4V;

                            s->mb_type[i] |= MB_TYPE_INTRA;

                        }

                    }

                    xy+=2;

                    i++;

                }

            }

        }

    }

}
