BlockBackend *blk_new_open(const char *filename, const char *reference,

                           QDict *options, int flags, Error **errp)

{

    BlockBackend *blk;

    BlockDriverState *bs;

    uint64_t perm;



    /* blk_new_open() is mainly used in .bdrv_create implementations and the

     * tools where sharing isn't a concern because the BDS stays private, so we

     * just request permission according to the flags.

     *

     * The exceptions are xen_disk and blockdev_init(); in these cases, the

     * caller of blk_new_open() doesn't make use of the permissions, but they

     * shouldn't hurt either. We can still share everything here because the

     * guest devices will add their own blockers if they can't share. */

    perm = BLK_PERM_CONSISTENT_READ;

    if (flags & BDRV_O_RDWR) {

        perm |= BLK_PERM_WRITE;

    }

    if (flags & BDRV_O_RESIZE) {

        perm |= BLK_PERM_RESIZE;

    }



    blk = blk_new(perm, BLK_PERM_ALL);

    bs = bdrv_open(filename, reference, options, flags, errp);

    if (!bs) {

        blk_unref(blk);

        return NULL;

    }



    blk->root = bdrv_root_attach_child(bs, "root", &child_root,

                                       perm, BLK_PERM_ALL, blk, &error_abort);



    return blk;

}
