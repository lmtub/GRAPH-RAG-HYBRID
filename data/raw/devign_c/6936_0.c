void commit_start(const char *job_id, BlockDriverState *bs,

                  BlockDriverState *base, BlockDriverState *top, int64_t speed,

                  BlockdevOnError on_error, const char *backing_file_str,

                  Error **errp)

{

    CommitBlockJob *s;

    BlockReopenQueue *reopen_queue = NULL;

    int orig_overlay_flags;

    int orig_base_flags;

    BlockDriverState *iter;

    BlockDriverState *overlay_bs;

    Error *local_err = NULL;

    int ret;



    assert(top != bs);

    if (top == base) {

        error_setg(errp, "Invalid files for merge: top and base are the same");

        return;

    }



    overlay_bs = bdrv_find_overlay(bs, top);



    if (overlay_bs == NULL) {

        error_setg(errp, "Could not find overlay image for %s:", top->filename);

        return;

    }



    /* FIXME Use real permissions */

    s = block_job_create(job_id, &commit_job_driver, bs, 0, BLK_PERM_ALL,

                         speed, BLOCK_JOB_DEFAULT, NULL, NULL, errp);

    if (!s) {

        return;

    }



    orig_base_flags    = bdrv_get_flags(base);

    orig_overlay_flags = bdrv_get_flags(overlay_bs);



    /* convert base & overlay_bs to r/w, if necessary */

    if (!(orig_base_flags & BDRV_O_RDWR)) {

        reopen_queue = bdrv_reopen_queue(reopen_queue, base, NULL,

                                         orig_base_flags | BDRV_O_RDWR);

    }

    if (!(orig_overlay_flags & BDRV_O_RDWR)) {

        reopen_queue = bdrv_reopen_queue(reopen_queue, overlay_bs, NULL,

                                         orig_overlay_flags | BDRV_O_RDWR);

    }

    if (reopen_queue) {

        bdrv_reopen_multiple(bdrv_get_aio_context(bs), reopen_queue, &local_err);

        if (local_err != NULL) {

            error_propagate(errp, local_err);

            goto fail;

        }

    }





    /* Block all nodes between top and base, because they will

     * disappear from the chain after this operation. */

    assert(bdrv_chain_contains(top, base));

    for (iter = top; iter != backing_bs(base); iter = backing_bs(iter)) {

        /* FIXME Use real permissions */

        block_job_add_bdrv(&s->common, "intermediate node", iter, 0,

                           BLK_PERM_ALL, &error_abort);

    }

    /* overlay_bs must be blocked because it needs to be modified to

     * update the backing image string, but if it's the root node then

     * don't block it again */

    if (bs != overlay_bs) {

        /* FIXME Use real permissions */

        block_job_add_bdrv(&s->common, "overlay of top", overlay_bs, 0,

                           BLK_PERM_ALL, &error_abort);

    }



    /* FIXME Use real permissions */

    s->base = blk_new(0, BLK_PERM_ALL);

    ret = blk_insert_bs(s->base, base, errp);

    if (ret < 0) {

        goto fail;

    }



    /* FIXME Use real permissions */

    s->top = blk_new(0, BLK_PERM_ALL);

    ret = blk_insert_bs(s->top, top, errp);

    if (ret < 0) {

        goto fail;

    }



    s->active = bs;



    s->base_flags          = orig_base_flags;

    s->orig_overlay_flags  = orig_overlay_flags;



    s->backing_file_str = g_strdup(backing_file_str);



    s->on_error = on_error;



    trace_commit_start(bs, base, top, s);

    block_job_start(&s->common);

    return;



fail:

    if (s->base) {

        blk_unref(s->base);

    }

    if (s->top) {

        blk_unref(s->top);

    }

    block_job_unref(&s->common);

}
