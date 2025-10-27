static void flash_sync_page(Flash *s, int page)

{

    int bdrv_sector, nb_sectors;

    QEMUIOVector iov;



    if (!s->bdrv || bdrv_is_read_only(s->bdrv)) {

        return;

    }



    bdrv_sector = (page * s->pi->page_size) / BDRV_SECTOR_SIZE;

    nb_sectors = DIV_ROUND_UP(s->pi->page_size, BDRV_SECTOR_SIZE);

    qemu_iovec_init(&iov, 1);

    qemu_iovec_add(&iov, s->storage + bdrv_sector * BDRV_SECTOR_SIZE,

                   nb_sectors * BDRV_SECTOR_SIZE);

    bdrv_aio_writev(s->bdrv, bdrv_sector, &iov, nb_sectors, bdrv_sync_complete,

                    NULL);

}
