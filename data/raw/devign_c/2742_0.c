static void sdram_set_bcr (uint32_t *bcrp, uint32_t bcr, int enabled)

{

    if (*bcrp & 0x00000001) {

        /* Unmap RAM */

#ifdef DEBUG_SDRAM

        printf("%s: unmap RAM area " TARGET_FMT_plx " " TARGET_FMT_lx "\n",

               __func__, sdram_base(*bcrp), sdram_size(*bcrp));

#endif

        cpu_register_physical_memory(sdram_base(*bcrp), sdram_size(*bcrp),

                                     IO_MEM_UNASSIGNED);

    }

    *bcrp = bcr & 0xFFDEE001;

    if (enabled && (bcr & 0x00000001)) {

#ifdef DEBUG_SDRAM

        printf("%s: Map RAM area " TARGET_FMT_plx " " TARGET_FMT_lx "\n",

               __func__, sdram_base(bcr), sdram_size(bcr));

#endif

        cpu_register_physical_memory(sdram_base(bcr), sdram_size(bcr),

                                     sdram_base(bcr) | IO_MEM_RAM);

    }

}
