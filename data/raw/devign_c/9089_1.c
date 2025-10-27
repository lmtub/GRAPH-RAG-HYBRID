static void mirror_start_job(const char *job_id, BlockDriverState *bs,

                             BlockDriverState *target, const char *replaces,

                             int64_t speed, uint32_t granularity,

                             int64_t buf_size,

                             BlockMirrorBackingMode backing_mode,

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



    if (buf_size < 0) {

        error_setg(errp, "Invalid parameter 'buf-size'");

        return;

    }



    if (buf_size == 0) {

        buf_size = DEFAULT_MIRROR_BUF_SIZE;

    }



    s = block_job_create(job_id, driver, bs, speed, cb, opaque, errp);

    if (!s) {

        return;

    }



    s->target = blk_new();

    blk_insert_bs(s->target, target);



    s->replaces = g_strdup(replaces);

    s->on_source_error = on_source_error;

    s->on_target_error = on_target_error;

    s->is_none_mode = is_none_mode;

    s->backing_mode = backing_mode;

    s->base = base;

    s->granularity = granularity;

    s->buf_size = ROUND_UP(buf_size, granularity);

    s->unmap = unmap;



    s->dirty_bitmap = bdrv_create_dirty_bitmap(bs, granularity, NULL, errp);

    if (!s->dirty_bitmap) {

        g_free(s->replaces);

        blk_unref(s->target);

        block_job_unref(&s->common);

        return;

    }



    bdrv_op_block_all(target, s->common.blocker);



    s->common.co = qemu_coroutine_create(mirror_run);

    trace_mirror_start(bs, s, s->common.co, opaque);

    qemu_coroutine_enter(s->common.co, s);

}
