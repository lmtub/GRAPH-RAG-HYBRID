static av_cold void x8_vlc_init(void){

    int i;

    int offset = 0;

    int sizeidx = 0;

    static const uint16_t sizes[8*4 + 8*2 + 2 + 4] = {

        576, 548, 582, 618, 546, 616, 560, 642,

        584, 582, 704, 664, 512, 544, 656, 640,

        512, 648, 582, 566, 532, 614, 596, 648,

        586, 552, 584, 590, 544, 578, 584, 624,



        528, 528, 526, 528, 536, 528, 526, 544,

        544, 512, 512, 528, 528, 544, 512, 544,



        128, 128, 128, 128, 128, 128};



    static VLC_TYPE table[28150][2];



#define  init_ac_vlc(dst,src) \

    dst.table = &table[offset]; \

    dst.table_allocated = sizes[sizeidx]; \

    offset += sizes[sizeidx++]; \

       init_vlc(&dst, \

              AC_VLC_BITS,77, \

              &src[1],4,2, \

              &src[0],4,2, \

              INIT_VLC_USE_NEW_STATIC)

//set ac tables

    for(i=0;i<8;i++){

        init_ac_vlc( j_ac_vlc[0][0][i], x8_ac0_highquant_table[i][0] );

        init_ac_vlc( j_ac_vlc[0][1][i], x8_ac1_highquant_table[i][0] );

        init_ac_vlc( j_ac_vlc[1][0][i], x8_ac0_lowquant_table [i][0] );

        init_ac_vlc( j_ac_vlc[1][1][i], x8_ac1_lowquant_table [i][0] );

    }

#undef init_ac_vlc



//set dc tables

#define init_dc_vlc(dst,src) \

    dst.table = &table[offset]; \

    dst.table_allocated = sizes[sizeidx]; \

    offset += sizes[sizeidx++]; \

        init_vlc(&dst, \

        DC_VLC_BITS,34, \

        &src[1],4,2, \

        &src[0],4,2, \

        INIT_VLC_USE_NEW_STATIC);

    for(i=0;i<8;i++){

        init_dc_vlc( j_dc_vlc[0][i], x8_dc_highquant_table[i][0]);

        init_dc_vlc( j_dc_vlc[1][i], x8_dc_lowquant_table [i][0]);

    }

#undef init_dc_vlc



//set orient tables

#define init_or_vlc(dst,src) \

    dst.table = &table[offset]; \

    dst.table_allocated = sizes[sizeidx]; \

    offset += sizes[sizeidx++]; \

    init_vlc(&dst, \

    OR_VLC_BITS,12, \

    &src[1],4,2, \

    &src[0],4,2, \

    INIT_VLC_USE_NEW_STATIC);

    for(i=0;i<2;i++){

        init_or_vlc( j_orient_vlc[0][i], x8_orient_highquant_table[i][0]);

    }

    for(i=0;i<4;i++){

        init_or_vlc( j_orient_vlc[1][i], x8_orient_lowquant_table [i][0])

    }

    if (offset != sizeof(table)/sizeof(VLC_TYPE)/2)

        av_log(NULL, AV_LOG_ERROR, "table size %i does not match needed %i\n", (int)(sizeof(table)/sizeof(VLC_TYPE)/2), offset);

}
