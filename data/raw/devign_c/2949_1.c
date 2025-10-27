static av_cold void init_vlcs(FourXContext *f)

{

    static VLC_TYPE table[8][32][2];

    int i;



    for (i = 0; i < 8; i++) {

        block_type_vlc[0][i].table           = table[i];

        block_type_vlc[0][i].table_allocated = 32;

        init_vlc(&block_type_vlc[0][i], BLOCK_TYPE_VLC_BITS, 7,

                 &block_type_tab[0][i][0][1], 2, 1,

                 &block_type_tab[0][i][0][0], 2, 1, INIT_VLC_USE_NEW_STATIC);

    }

}
