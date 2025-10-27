SDState *sd_init(BlockDriverState *bs, bool is_spi)

{

    SDState *sd;



    if (bs && bdrv_is_read_only(bs)) {

        fprintf(stderr, "sd_init: Cannot use read-only drive\n");

        return NULL;

    }



    sd = (SDState *) g_malloc0(sizeof(SDState));

    sd->buf = qemu_blockalign(bs, 512);

    sd->spi = is_spi;

    sd->enable = true;

    sd_reset(sd, bs);

    if (sd->bdrv) {

        bdrv_attach_dev_nofail(sd->bdrv, sd);

        bdrv_set_dev_ops(sd->bdrv, &sd_block_ops, sd);

    }

    vmstate_register(NULL, -1, &sd_vmstate, sd);

    return sd;

}
