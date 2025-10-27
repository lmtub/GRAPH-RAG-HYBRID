void ff_fix_long_b_mvs(MpegEncContext * s, int16_t (*mv_table)[2], int f_code, int type)

{

    int y;

    uint8_t * fcode_tab= s->fcode_tab;



    // RAL: 8 in MPEG-1, 16 in MPEG-4

    int range = (((s->codec_id == CODEC_ID_MPEG1VIDEO) ? 8 : 16) << f_code);



    /* clip / convert to intra 16x16 type MVs */

    for(y=0; y<s->mb_height; y++){

        int x;

        int xy= (y+1)* (s->mb_width+2)+1;

        int i= y*s->mb_width;

        for(x=0; x<s->mb_width; x++)

            {

            if (s->mb_type[i] & type)    // RAL: "type" test added...

                {

                if (fcode_tab[mv_table[xy][0] + MAX_MV] > f_code || fcode_tab[mv_table[xy][0] + MAX_MV] == 0)

                    {

                    if(mv_table[xy][0]>0) 

                        mv_table[xy][0]=  range-1;

                    else

                        mv_table[xy][0]= -range;

                    }

                if (fcode_tab[mv_table[xy][1] + MAX_MV] > f_code || fcode_tab[mv_table[xy][1] + MAX_MV] == 0)

                    {

                    if(mv_table[xy][1]>0) 

                        mv_table[xy][1]=  range-1;

                    else                  

                        mv_table[xy][1]= -range;

            }

            }

            xy++;

            i++;

        }

    }

}
