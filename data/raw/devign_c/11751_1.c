static void commit_complete(BlockJob *job, void *opaque)
{
    CommitBlockJob *s = container_of(job, CommitBlockJob, common);
    CommitCompleteData *data = opaque;
    BlockDriverState *active = s->active;
    BlockDriverState *top = blk_bs(s->top);
    BlockDriverState *base = blk_bs(s->base);
    BlockDriverState *overlay_bs = bdrv_find_overlay(active, s->commit_top_bs);
    int ret = data->ret;
    bool remove_commit_top_bs = false;
    /* Make sure overlay_bs and top stay around until bdrv_set_backing_hd() */
    bdrv_ref(top);
    bdrv_ref(overlay_bs);
    /* Remove base node parent that still uses BLK_PERM_WRITE/RESIZE before
     * the normal backing chain can be restored. */
    blk_unref(s->base);
    if (!block_job_is_cancelled(&s->common) && ret == 0) {
        /* success */
        ret = bdrv_drop_intermediate(active, s->commit_top_bs, base,
                                     s->backing_file_str);
    } else if (overlay_bs) {
        /* XXX Can (or should) we somehow keep 'consistent read' blocked even
         * after the failed/cancelled commit job is gone? If we already wrote
         * something to base, the intermediate images aren't valid any more. */
        remove_commit_top_bs = true;
    }
    /* restore base open flags here if appropriate (e.g., change the base back
     * to r/o). These reopens do not need to be atomic, since we won't abort
     * even on failure here */
    if (s->base_flags != bdrv_get_flags(base)) {
        bdrv_reopen(base, s->base_flags, NULL);
    }
    if (overlay_bs && s->orig_overlay_flags != bdrv_get_flags(overlay_bs)) {
        bdrv_reopen(overlay_bs, s->orig_overlay_flags, NULL);
    }
    g_free(s->backing_file_str);
    blk_unref(s->top);
    block_job_completed(&s->common, ret);
    g_free(data);
    /* If bdrv_drop_intermediate() didn't already do that, remove the commit
     * filter driver from the backing chain. Do this as the final step so that
     * the 'consistent read' permission can be granted.  */
    if (remove_commit_top_bs) {
        bdrv_set_backing_hd(overlay_bs, top, &error_abort);
    }
}