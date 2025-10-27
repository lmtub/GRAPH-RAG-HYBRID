av_cold void ff_msmpeg4_encode_init(MpegEncContext *s)

{

    static int init_done=0;

    int i;



    ff_msmpeg4_common_init(s);

    if(s->msmpeg4_version>=4){

        s->min_qcoeff= -255;

        s->max_qcoeff=  255;

    }



    if (!init_done) {

        /* init various encoding tables */

        init_done = 1;

        init_mv_table(&ff_mv_tables[0]);

        init_mv_table(&ff_mv_tables[1]);

        for(i=0;i<NB_RL_TABLES;i++)

            ff_init_rl(&ff_rl_table[i], ff_static_rl_table_store[i]);



        for(i=0; i<NB_RL_TABLES; i++){

            int level;

            for (level = 1; level <= MAX_LEVEL; level++) {

                int run;

                for(run=0; run<=MAX_RUN; run++){

                    int last;

                    for(last=0; last<2; last++){

                        rl_length[i][level][run][last]= get_size_of_code(s, &ff_rl_table[  i], last, run, level, 0);

                    }

                }

            }

        }

    }

}
