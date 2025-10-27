static void spapr_nvram_realize(VIOsPAPRDevice *dev, Error **errp)

{

    sPAPRNVRAM *nvram = VIO_SPAPR_NVRAM(dev);

    int ret;



    if (nvram->blk) {

        nvram->size = blk_getlength(nvram->blk);



        ret = blk_set_perm(nvram->blk,

                           BLK_PERM_CONSISTENT_READ | BLK_PERM_WRITE,

                           BLK_PERM_ALL, errp);

        if (ret < 0) {

            return;

        }

    } else {

        nvram->size = DEFAULT_NVRAM_SIZE;

    }



    nvram->buf = g_malloc0(nvram->size);



    if ((nvram->size < MIN_NVRAM_SIZE) || (nvram->size > MAX_NVRAM_SIZE)) {

        error_setg(errp, "spapr-nvram must be between %d and %d bytes in size",

                   MIN_NVRAM_SIZE, MAX_NVRAM_SIZE);

        return;

    }



    if (nvram->blk) {

        int alen = blk_pread(nvram->blk, 0, nvram->buf, nvram->size);



        if (alen != nvram->size) {

            error_setg(errp, "can't read spapr-nvram contents");

            return;

        }

    } else if (nb_prom_envs > 0) {

        /* Create a system partition to pass the -prom-env variables */

        chrp_nvram_create_system_partition(nvram->buf, MIN_NVRAM_SIZE / 4);

        chrp_nvram_create_free_partition(&nvram->buf[MIN_NVRAM_SIZE / 4],

                                         nvram->size - MIN_NVRAM_SIZE / 4);

    }



    spapr_rtas_register(RTAS_NVRAM_FETCH, "nvram-fetch", rtas_nvram_fetch);

    spapr_rtas_register(RTAS_NVRAM_STORE, "nvram-store", rtas_nvram_store);

}
