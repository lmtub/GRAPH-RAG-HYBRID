void qmp_cont(Error **errp)

{

    Error *local_err = NULL;

    BlockBackend *blk;

    BlockDriverState *bs;

    BdrvNextIterator it;



    /* if there is a dump in background, we should wait until the dump

     * finished */

    if (dump_in_progress()) {

        error_setg(errp, "There is a dump in process, please wait.");

        return;

    }



    if (runstate_needs_reset()) {

        error_setg(errp, "Resetting the Virtual Machine is required");

        return;

    } else if (runstate_check(RUN_STATE_SUSPENDED)) {

        return;

    }



    for (blk = blk_next(NULL); blk; blk = blk_next(blk)) {

        blk_iostatus_reset(blk);

    }



    for (bs = bdrv_first(&it); bs; bs = bdrv_next(&it)) {

        bdrv_add_key(bs, NULL, &local_err);

        if (local_err) {

            error_propagate(errp, local_err);

            return;

        }

    }



    /* Continuing after completed migration. Images have been inactivated to

     * allow the destination to take control. Need to get control back now.

     *

     * If there are no inactive block nodes (e.g. because the VM was just

     * paused rather than completing a migration), bdrv_inactivate_all() simply

     * doesn't do anything. */

    bdrv_invalidate_cache_all(&local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    blk_resume_after_migration(&local_err);

    if (local_err) {

        error_propagate(errp, local_err);

        return;

    }



    if (runstate_check(RUN_STATE_INMIGRATE)) {

        autostart = 1;

    } else {

        vm_start();

    }

}
