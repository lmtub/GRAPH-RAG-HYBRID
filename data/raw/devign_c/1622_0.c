static void mirror_start_job(BlockDriverState *bs, BlockDriverState *target,

                             const char *replaces,

                             int64_t speed, uint32_t granularity,

                             int64_t buf_size,

                             BlockdevOnError on_source_error,

                             BlockdevOnError on_target_error,

                             BlockCompletionFunc *cb,

                             void *opaque, Error **errp,

                             const BlockJobDriver *driver,

                             bool is_none_mode, BlockDriverState *base)

{

    MirrorBlockJob *s;



    if (granularity == 0) {

        /* Choose the default granularity based on the target file's cluster

         * size, clamped between 4k and 64k.  */

        BlockDriverInfo bdi;

        if (bdrv_get_info(target, &bdi) >= 0 && bdi.cluster_size != 0) {

            granularity = MAX(4096, bdi.cluster_size);

            granularity = MIN(65536, granularity);

        } else {

            granularity = 65536;

        }

    }



    assert ((granularity & (granularity - 1)) == 0);



    if ((on_source_error == BLOCKDEV_ON_ERROR_STOP ||

         on_source_error == BLOCKDEV_ON_ERROR_ENOSPC) &&

        !bdrv_iostatus_is_enabled(bs)) {

        error_set(errp, QERR_INVALID_PARAMETER, "on-source-error");

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

    s->buf_size = MAX(buf_size, granularity);



    s->dirty_bitmap = bdrv_create_dirty_bitmap(bs, granularity, NULL, errp);

    if (!s->dirty_bitmap) {

        return;

    }

    bdrv_set_enable_write_cache(s->target, true);

    bdrv_set_on_error(s->target, on_target_error, on_target_error);

    bdrv_iostatus_enable(s->target);

    s->common.co = qemu_coroutine_create(mirror_run);

    trace_mirror_start(bs, s, s->common.co, opaque);

    qemu_coroutine_enter(s->common.co, s);

}
