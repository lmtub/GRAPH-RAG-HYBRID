NBDExport *nbd_export_new(BlockBackend *blk, off_t dev_offset, off_t size,

                          uint32_t nbdflags, void (*close)(NBDExport *),

                          Error **errp)

{

    NBDExport *exp = g_malloc0(sizeof(NBDExport));

    exp->refcount = 1;

    QTAILQ_INIT(&exp->clients);

    exp->blk = blk;

    exp->dev_offset = dev_offset;

    exp->nbdflags = nbdflags;

    exp->size = size < 0 ? blk_getlength(blk) : size;

    if (exp->size < 0) {

        error_setg_errno(errp, -exp->size,

                         "Failed to determine the NBD export's length");

        goto fail;

    }

    exp->size -= exp->size % BDRV_SECTOR_SIZE;



    exp->close = close;

    exp->ctx = blk_get_aio_context(blk);

    blk_ref(blk);

    blk_add_aio_context_notifier(blk, blk_aio_attached, blk_aio_detach, exp);



    exp->eject_notifier.notify = nbd_eject_notifier;

    blk_add_remove_bs_notifier(blk, &exp->eject_notifier);



    /*

     * NBD exports are used for non-shared storage migration.  Make sure

     * that BDRV_O_INACTIVE is cleared and the image is ready for write

     * access since the export could be available before migration handover.

     */

    aio_context_acquire(exp->ctx);

    blk_invalidate_cache(blk, NULL);

    aio_context_release(exp->ctx);

    return exp;



fail:

    g_free(exp);

    return NULL;

}
