void blk_io_limits_disable(BlockBackend *blk)

{

    assert(blk->public.throttle_group_member.throttle_state);

    bdrv_drained_begin(blk_bs(blk));

    throttle_group_unregister_tgm(&blk->public.throttle_group_member);

    bdrv_drained_end(blk_bs(blk));

}
