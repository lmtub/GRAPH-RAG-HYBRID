static void mirror_exit(BlockJob *job, void *opaque)

{

    MirrorBlockJob *s = container_of(job, MirrorBlockJob, common);

    MirrorExitData *data = opaque;

    AioContext *replace_aio_context = NULL;

    BlockDriverState *src = blk_bs(s->common.blk);

    BlockDriverState *target_bs = blk_bs(s->target);



    /* Make sure that the source BDS doesn't go away before we called

     * block_job_completed(). */

    bdrv_ref(src);



    if (s->to_replace) {

        replace_aio_context = bdrv_get_aio_context(s->to_replace);

        aio_context_acquire(replace_aio_context);

    }



    if (s->should_complete && data->ret == 0) {

        BlockDriverState *to_replace = src;

        if (s->to_replace) {

            to_replace = s->to_replace;

        }



        if (bdrv_get_flags(target_bs) != bdrv_get_flags(to_replace)) {

            bdrv_reopen(target_bs, bdrv_get_flags(to_replace), NULL);

        }



        /* The mirror job has no requests in flight any more, but we need to

         * drain potential other users of the BDS before changing the graph. */

        bdrv_drained_begin(target_bs);

        bdrv_replace_in_backing_chain(to_replace, target_bs);

        bdrv_drained_end(target_bs);



        /* We just changed the BDS the job BB refers to */

        blk_remove_bs(job->blk);

        blk_insert_bs(job->blk, src);

    }

    if (s->to_replace) {

        bdrv_op_unblock_all(s->to_replace, s->replace_blocker);

        error_free(s->replace_blocker);

        bdrv_unref(s->to_replace);

    }

    if (replace_aio_context) {

        aio_context_release(replace_aio_context);

    }

    g_free(s->replaces);

    bdrv_op_unblock_all(target_bs, s->common.blocker);

    blk_unref(s->target);

    block_job_completed(&s->common, data->ret);

    g_free(data);

    bdrv_drained_end(src);

    if (qemu_get_aio_context() == bdrv_get_aio_context(src)) {

        aio_enable_external(iohandler_get_aio_context());

    }

    bdrv_unref(src);

}
