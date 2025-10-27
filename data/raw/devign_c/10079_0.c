static void find_best_tables(MpegEncContext * s)

{

    int i;

    int best       =-1, best_size       =9999999;

    int chroma_best=-1, best_chroma_size=9999999;



    for(i=0; i<3; i++){

        int level;

        int chroma_size=0;

        int size=0;



        if(i>0){// ;)

            size++;

            chroma_size++;

        }

        for(level=0; level<=MAX_LEVEL; level++){

            int run;

            for(run=0; run<=MAX_RUN; run++){

                int last;

                const int last_size= size + chroma_size;

                for(last=0; last<2; last++){

                    int inter_count       = s->ac_stats[0][0][level][run][last] + s->ac_stats[0][1][level][run][last];

                    int intra_luma_count  = s->ac_stats[1][0][level][run][last];

                    int intra_chroma_count= s->ac_stats[1][1][level][run][last];



                    if(s->pict_type==AV_PICTURE_TYPE_I){

                        size       += intra_luma_count  *rl_length[i  ][level][run][last];

                        chroma_size+= intra_chroma_count*rl_length[i+3][level][run][last];

                    }else{

                        size+=        intra_luma_count  *rl_length[i  ][level][run][last]

                                     +intra_chroma_count*rl_length[i+3][level][run][last]

                                     +inter_count       *rl_length[i+3][level][run][last];

                    }

                }

                if(last_size == size+chroma_size) break;

            }

        }

        if(size<best_size){

            best_size= size;

            best= i;

        }

        if(chroma_size<best_chroma_size){

            best_chroma_size= chroma_size;

            chroma_best= i;

        }

    }



//    printf("type:%d, best:%d, qp:%d, var:%d, mcvar:%d, size:%d //\n",

//           s->pict_type, best, s->qscale, s->mb_var_sum, s->mc_mb_var_sum, best_size);



    if(s->pict_type==AV_PICTURE_TYPE_P) chroma_best= best;



    memset(s->ac_stats, 0, sizeof(int)*(MAX_LEVEL+1)*(MAX_RUN+1)*2*2*2);



    s->rl_table_index       =        best;

    s->rl_chroma_table_index= chroma_best;



    if(s->pict_type != s->last_non_b_pict_type){

        s->rl_table_index= 2;

        if(s->pict_type==AV_PICTURE_TYPE_I)

            s->rl_chroma_table_index= 1;

        else

            s->rl_chroma_table_index= 2;

    }



}
