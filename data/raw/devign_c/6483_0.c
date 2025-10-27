int ff_get_best_fcode(MpegEncContext * s, int16_t (*mv_table)[2], int type)

{

    int f_code;



    if(s->me_method>=ME_EPZS){

        int mv_num[8];

        int i, y;

        int loose=0;

        UINT8 * fcode_tab= s->fcode_tab;



        for(i=0; i<8; i++) mv_num[i]=0;



        for(y=0; y<s->mb_height; y++){

            int x;

            int xy= (y+1)* (s->mb_width+2) + 1;

            i= y*s->mb_width;

            for(x=0; x<s->mb_width; x++){

                if(s->mb_type[i] & type){

                    mv_num[ fcode_tab[mv_table[xy][0] + MAX_MV] ]++;

                    mv_num[ fcode_tab[mv_table[xy][1] + MAX_MV] ]++;

//printf("%d %d %d\n", s->mv_table[0][i], fcode_tab[s->mv_table[0][i] + MAX_MV], i);

                }

                i++;

                xy++;

            }

        }



        for(i=MAX_FCODE; i>1; i--){

            int threshold;

            loose+= mv_num[i];



            if(s->pict_type==B_TYPE) threshold= 0;

            else                     threshold= s->mb_num/20; //FIXME 

            if(loose > threshold) break;

        }

//    printf("fcode: %d type: %d\n", i, s->pict_type);

        return i;

/*        for(i=0; i<=MAX_FCODE; i++){

            printf("%d ", mv_num[i]);

        }

        printf("\n");*/

    }else{

        return 1;

    }

}
