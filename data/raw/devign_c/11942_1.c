static void init_vlcs(FourXContext *f){

    static int done = 0;

    int i;



    if (!done) {

        done = 1;



        for(i=0; i<4; i++){

            init_vlc(&block_type_vlc[i], BLOCK_TYPE_VLC_BITS, 7, 

                     &block_type_tab[i][0][1], 2, 1,

                     &block_type_tab[i][0][0], 2, 1);

        }

    }

}
