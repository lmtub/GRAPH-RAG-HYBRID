void commit_start(BlockDriverState *bs, BlockDriverState *base,

                  BlockDriverState *top, int64_t speed,

                  BlockdevOnError on_error, BlockDriverCompletionFunc *cb,

                  void *opaque, Error **errp)

{

    CommitBlockJob *s;

    BlockReopenQueue *reopen_queue = NULL;

    int orig_overlay_flags;

    int orig_base_flags;

    BlockDriverState *overlay_bs;

    Error *local_err = NULL;



    if ((on_error == BLOCKDEV_ON_ERROR_STOP ||

         on_error == BLOCKDEV_ON_ERROR_ENOSPC) &&

        !bdrv_iostatus_is_enabled(bs)) {

        error_set(errp, QERR_INVALID_PARAMETER_COMBINATION);

        return;

    }



    /* Once we support top == active layer, remove this check */

    if (top == bs) {

        error_setg(errp,

                   "Top image as the active layer is currently unsupported");

        return;

    }



    if (top == base) {

        error_setg(errp, "Invalid files for merge: top and base are the same");

        return;

    }



    /* top and base may be valid, but let's make sure that base is reachable

     * from top */

    if (bdrv_find_backing_image(top, base->filename) != base) {

        error_setg(errp,

                   "Base (%s) is not reachable from top (%s)",

                   base->filename, top->filename);

        return;

    }



    overlay_bs = bdrv_find_overlay(bs, top);



    if (overlay_bs == NULL) {

        error_setg(errp, "Could not find overlay image for %s:", top->filename);

        return;

    }



    orig_base_flags    = bdrv_get_flags(base);

    orig_overlay_flags = bdrv_get_flags(overlay_bs);



    /* convert base & overlay_bs to r/w, if necessary */

    if (!(orig_base_flags & BDRV_O_RDWR)) {

        reopen_queue = bdrv_reopen_queue(reopen_queue, base,

                                         orig_base_flags | BDRV_O_RDWR);

    }

    if (!(orig_overlay_flags & BDRV_O_RDWR)) {

        reopen_queue = bdrv_reopen_queue(reopen_queue, overlay_bs,

                                         orig_overlay_flags | BDRV_O_RDWR);

    }

    if (reopen_queue) {

        bdrv_reopen_multiple(reopen_queue, &local_err);

        if (local_err != NULL) {

            error_propagate(errp, local_err);

            return;

        }

    }





    s = block_job_create(&commit_job_type, bs, speed, cb, opaque, errp);

    if (!s) {

        return;

    }



    s->base   = base;

    s->top    = top;

    s->active = bs;



    s->base_flags          = orig_base_flags;

    s->orig_overlay_flags  = orig_overlay_flags;



    s->on_error = on_error;

    s->common.co = qemu_coroutine_create(commit_run);



    trace_commit_start(bs, base, top, s, s->common.co, opaque);

    qemu_coroutine_enter(s->common.co, s);

}
