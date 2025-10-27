static void mirror_start_job(BlockDriverState *bs, BlockDriverState *target,

                             const char *replaces,

                             int64_t speed, uint32_t granularity,

                             int64_t buf_size,

                             BlockdevOnError on_source_error,

                             BlockdevOnError on_target_error,

                             bool unmap,

                             BlockCompletionFunc *cb,

                             void *opaque, Error **errp,

                             const BlockJobDriver *driver,

                             bool is_none_mode, BlockDriverState *base)

{

    MirrorBlockJob *s;




    if (granularity == 0) {

        granularity = bdrv_get_default_bitmap_granularity(target);

    }



    assert ((granularity & (granularity - 1)) == 0);



    if ((on_source_error == BLOCKDEV_ON_ERROR_STOP ||

         on_source_error == BLOCKDEV_ON_ERROR_ENOSPC) &&

        (!bs->blk || !blk_iostatus_is_enabled(bs->blk))) {

        error_setg(errp, QERR_INVALID_PARAMETER, "on-source-error");

        return;

    }



    if (buf_size < 0) {

        error_setg(errp, "Invalid parameter 'buf-size'");

        return;

    }



    if (buf_size == 0) {

        buf_size = DEFAULT_MIRROR_BUF_SIZE;

    }



    /* We can't support this case as long as the block layer can't handle

     * multiple BlockBackends per BlockDriverState. */

    if (replaces) {

        replaced_bs = bdrv_lookup_bs(replaces, replaces, errp);

        if (replaced_bs == NULL) {

            return;

        }

    } else {

        replaced_bs = bs;

    }

    if (replaced_bs->blk && target->blk) {

        error_setg(errp, "Can't create node with two BlockBackends");

        return;

    }



    s = block_job_create(driver, bs, speed, cb, opaque, errp);

    if (!s) {

        return;

    }



    s->replaces = g_strdup(replaces);

    s->on_source_error = on_source_error;

    s->on_target_error = on_target_error;

    s->target = target;

    s->is_none_mode = is_none_mode;

    s->base = base;

    s->granularity = granularity;

    s->buf_size = ROUND_UP(buf_size, granularity);

    s->unmap = unmap;



    s->dirty_bitmap = bdrv_create_dirty_bitmap(bs, granularity, NULL, errp);

    if (!s->dirty_bitmap) {

        g_free(s->replaces);

        block_job_unref(&s->common);

        return;

    }



    bdrv_op_block_all(s->target, s->common.blocker);



    bdrv_set_enable_write_cache(s->target, true);

    if (s->target->blk) {

        blk_set_on_error(s->target->blk, on_target_error, on_target_error);

        blk_iostatus_enable(s->target->blk);

    }

    s->common.co = qemu_coroutine_create(mirror_run);

    trace_mirror_start(bs, s, s->common.co, opaque);

    qemu_coroutine_enter(s->common.co, s);

}