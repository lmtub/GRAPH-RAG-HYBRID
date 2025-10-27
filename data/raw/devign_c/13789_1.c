void blk_remove_bs(BlockBackend *blk)

{

    BlockDriverState *bs;

    ThrottleTimers *tt;



    notifier_list_notify(&blk->remove_bs_notifiers, blk);

    if (blk->public.throttle_group_member.throttle_state) {

        tt = &blk->public.throttle_group_member.throttle_timers;

        bs = blk_bs(blk);

        bdrv_drained_begin(bs);

        throttle_timers_detach_aio_context(tt);

        bdrv_drained_end(bs);

    }



    blk_update_root_state(blk);



    bdrv_root_unref_child(blk->root);

    blk->root = NULL;

}
