static int spapr_nvram_init(VIOsPAPRDevice *dev)

{

    sPAPRNVRAM *nvram = VIO_SPAPR_NVRAM(dev);



    if (nvram->drive) {

        nvram->size = bdrv_getlength(nvram->drive);

    } else {

        nvram->size = DEFAULT_NVRAM_SIZE;

        nvram->buf = g_malloc0(nvram->size);

    }



    if ((nvram->size < MIN_NVRAM_SIZE) || (nvram->size > MAX_NVRAM_SIZE)) {

        fprintf(stderr, "spapr-nvram must be between %d and %d bytes in size\n",

                MIN_NVRAM_SIZE, MAX_NVRAM_SIZE);

        return -1;

    }



    spapr_rtas_register("nvram-fetch", rtas_nvram_fetch);

    spapr_rtas_register("nvram-store", rtas_nvram_store);



    return 0;

}
