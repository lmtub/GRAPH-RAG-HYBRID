static av_cold void init_static(void)

{

    if (!huff_vlc[0].bits) {

    INIT_VLC_STATIC(&huff_vlc[0], VLC_BITS, 18,

                &ff_mlp_huffman_tables[0][0][1], 2, 1,

                &ff_mlp_huffman_tables[0][0][0], 2, 1, 512);

    INIT_VLC_STATIC(&huff_vlc[1], VLC_BITS, 16,

                &ff_mlp_huffman_tables[1][0][1], 2, 1,

                &ff_mlp_huffman_tables[1][0][0], 2, 1, 512);

    INIT_VLC_STATIC(&huff_vlc[2], VLC_BITS, 15,

                &ff_mlp_huffman_tables[2][0][1], 2, 1,

                &ff_mlp_huffman_tables[2][0][0], 2, 1, 512);




    ff_mlp_init_crc();
