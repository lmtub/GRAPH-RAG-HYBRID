static inline void flash_sync_area(Flash *s, int64_t off, int64_t len)

{

    QEMUIOVector *iov = g_new(QEMUIOVector, 1);



    if (!s->blk || blk_is_read_only(s->blk)) {

        return;

    }



    assert(!(len % BDRV_SECTOR_SIZE));

    qemu_iovec_init(iov, 1);

    qemu_iovec_add(iov, s->storage + off, len);

    blk_aio_pwritev(s->blk, off, iov, 0, blk_sync_complete, iov);

}
