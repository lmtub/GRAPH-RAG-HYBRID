static void sdram_map_bcr (ppc4xx_sdram_t *sdram)

{

    int i;



    for (i = 0; i < sdram->nbanks; i++) {

        if (sdram->ram_sizes[i] != 0) {

            sdram_set_bcr(&sdram->bcr[i],

                          sdram_bcr(sdram->ram_bases[i], sdram->ram_sizes[i]),

                          1);

        } else {

            sdram_set_bcr(&sdram->bcr[i], 0x00000000, 0);

        }

    }

}
