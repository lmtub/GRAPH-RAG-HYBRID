void commit_start(const char *job_id, BlockDriverState *bs,

                  BlockDriverState *base, BlockDriverState *top, int64_t speed,

                  BlockdevOnError on_error, BlockCompletionFunc *cb,

                  void *opaque, const char *backing_file_str, Error **errp)

{

    CommitBlockJob *s;

    BlockReopenQueue *reopen_queue = NULL;

    int orig_overlay_flags;

    int orig_base_flags;

    BlockDriverState *overlay_bs;

    Error *local_err = NULL;



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



    s = block_job_create(job_id, &commit_job_driver, bs, speed,

                         cb, opaque, errp);

    if (!s) {

        return;

    }



    orig_base_flags    = bdrv_get_flags(base);

    orig_overlay_flags = bdrv_get_flags(overlay_bs);



    /* convert base & overlay_bs to r/w, if necessary */

    if (!(orig_overlay_flags & BDRV_O_RDWR)) {

        reopen_queue = bdrv_reopen_queue(reopen_queue, overlay_bs, NULL,

                                         orig_overlay_flags | BDRV_O_RDWR);

    }

    if (!(orig_base_flags & BDRV_O_RDWR)) {

        reopen_queue = bdrv_reopen_queue(reopen_queue, base, NULL,

                                         orig_base_flags | BDRV_O_RDWR);

    }

    if (reopen_queue) {

        bdrv_reopen_multiple(reopen_queue, &local_err);

        if (local_err != NULL) {

            error_propagate(errp, local_err);

            block_job_unref(&s->common);

            return;

        }

    }





    s->base = blk_new();

    blk_insert_bs(s->base, base);



    s->top = blk_new();

    blk_insert_bs(s->top, top);



    s->active = bs;



    s->base_flags          = orig_base_flags;

    s->orig_overlay_flags  = orig_overlay_flags;



    s->backing_file_str = g_strdup(backing_file_str);



    s->on_error = on_error;

    s->common.co = qemu_coroutine_create(commit_run);



    trace_commit_start(bs, base, top, s, s->common.co, opaque);

    qemu_coroutine_enter(s->common.co, s);

}
