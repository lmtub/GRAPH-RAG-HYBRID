static int init_ralf_vlc(VLC *vlc, const uint8_t *data, int elems)

{

    uint8_t  lens[MAX_ELEMS];

    uint16_t codes[MAX_ELEMS];

    int counts[17], prefixes[18];

    int i, cur_len;

    int max_bits = 0;

    GetBitContext gb;



    init_get_bits(&gb, data, elems * 4);



    for (i = 0; i <= 16; i++)

        counts[i] = 0;

    for (i = 0; i < elems; i++) {

        cur_len  = get_bits(&gb, 4) + 1;

        counts[cur_len]++;

        max_bits = FFMAX(max_bits, cur_len);

        lens[i]  = cur_len;

    }

    prefixes[1] = 0;

    for (i = 1; i <= 16; i++)

        prefixes[i + 1] = (prefixes[i] + counts[i]) << 1;



    for (i = 0; i < elems; i++)

        codes[i] = prefixes[lens[i]]++;



    return ff_init_vlc_sparse(vlc, FFMIN(max_bits, 9), elems,

                              lens, 1, 1, codes, 2, 2, NULL, 0, 0, 0);

}
