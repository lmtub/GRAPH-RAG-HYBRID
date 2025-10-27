static void sdram_unmap_bcr (ppc4xx_sdram_t *sdram)

{

    int i;



    for (i = 0; i < sdram->nbanks; i++) {

#ifdef DEBUG_SDRAM

        printf("%s: Unmap RAM area " TARGET_FMT_plx " " TARGET_FMT_lx "\n",

               __func__, sdram_base(sdram->bcr[i]), sdram_size(sdram->bcr[i]));

#endif

        cpu_register_physical_memory(sdram_base(sdram->bcr[i]),

                                     sdram_size(sdram->bcr[i]),

                                     IO_MEM_UNASSIGNED);

    }

}
