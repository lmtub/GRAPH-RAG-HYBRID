void sd_set_cb(SDState *sd, qemu_irq readonly, qemu_irq insert)

{

    sd->readonly_cb = readonly;

    sd->inserted_cb = insert;

    qemu_set_irq(readonly, sd->bdrv ? bdrv_is_read_only(sd->bdrv) : 0);

    qemu_set_irq(insert, sd->bdrv ? bdrv_is_inserted(sd->bdrv) : 0);

}
