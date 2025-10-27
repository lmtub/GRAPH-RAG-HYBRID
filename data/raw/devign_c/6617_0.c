static void dca_init_vlcs(void)

{

    static int vlcs_inited = 0;

    int i, j;



    if (vlcs_inited)

        return;



    dca_bitalloc_index.offset = 1;

    dca_bitalloc_index.wrap = 2;

    for (i = 0; i < 5; i++)

        init_vlc(&dca_bitalloc_index.vlc[i], bitalloc_12_vlc_bits[i], 12,

                 bitalloc_12_bits[i], 1, 1,

                 bitalloc_12_codes[i], 2, 2, 1);

    dca_scalefactor.offset = -64;

    dca_scalefactor.wrap = 2;

    for (i = 0; i < 5; i++)

        init_vlc(&dca_scalefactor.vlc[i], SCALES_VLC_BITS, 129,

                 scales_bits[i], 1, 1,

                 scales_codes[i], 2, 2, 1);

    dca_tmode.offset = 0;

    dca_tmode.wrap = 1;

    for (i = 0; i < 4; i++)

        init_vlc(&dca_tmode.vlc[i], tmode_vlc_bits[i], 4,

                 tmode_bits[i], 1, 1,

                 tmode_codes[i], 2, 2, 1);



    for(i = 0; i < 10; i++)

        for(j = 0; j < 7; j++){

            if(!bitalloc_codes[i][j]) break;

            dca_smpl_bitalloc[i+1].offset = bitalloc_offsets[i];

            dca_smpl_bitalloc[i+1].wrap = 1 + (j > 4);

            init_vlc(&dca_smpl_bitalloc[i+1].vlc[j], bitalloc_maxbits[i][j],

                     bitalloc_sizes[i],

                     bitalloc_bits[i][j], 1, 1,

                     bitalloc_codes[i][j], 2, 2, 1);

        }

    vlcs_inited = 1;

}
