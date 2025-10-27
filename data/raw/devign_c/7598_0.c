static av_cold void init_coef_vlc(VLC *vlc, uint16_t **prun_table,

                                  float **plevel_table, uint16_t **pint_table,

                                  const CoefVLCTable *vlc_table)

{

    int n                        = vlc_table->n;

    const uint8_t  *table_bits   = vlc_table->huffbits;

    const uint32_t *table_codes  = vlc_table->huffcodes;

    const uint16_t *levels_table = vlc_table->levels;

    uint16_t *run_table, *level_table, *int_table;

    float *flevel_table;

    int i, l, j, k, level;



    init_vlc(vlc, VLCBITS, n, table_bits, 1, 1, table_codes, 4, 4, 0);



    run_table    = av_malloc(n * sizeof(uint16_t));

    level_table  = av_malloc(n * sizeof(uint16_t));

    flevel_table = av_malloc(n * sizeof(*flevel_table));

    int_table    = av_malloc(n * sizeof(uint16_t));

    i            = 2;

    level        = 1;

    k            = 0;

    while (i < n) {

        int_table[k] = i;

        l            = levels_table[k++];

        for (j = 0; j < l; j++) {

            run_table[i]    = j;

            level_table[i]  = level;

            flevel_table[i] = level;

            i++;

        }

        level++;

    }

    *prun_table   = run_table;

    *plevel_table = flevel_table;

    *pint_table   = int_table;

    av_free(level_table);

}
