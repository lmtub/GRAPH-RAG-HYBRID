static void vp6_build_huff_tree(VP56Context *s, uint8_t coeff_model[],

                                const uint8_t *map, unsigned size, VLC *vlc)

{

    Node nodes[2*size], *tmp = &nodes[size];

    int a, b, i;



    /* first compute probabilities from model */

    tmp[0].count = 256;

    for (i=0; i<size-1; i++) {

        a = tmp[i].count *        coeff_model[i]  >> 8;

        b = tmp[i].count * (255 - coeff_model[i]) >> 8;

        nodes[map[2*i  ]].count = a + !a;

        nodes[map[2*i+1]].count = b + !b;

    }




    /* then build the huffman tree accodring to probabilities */

    ff_huff_build_tree(s->avctx, vlc, size, nodes, vp6_huff_cmp,

                       FF_HUFFMAN_FLAG_HNODE_FIRST);

}