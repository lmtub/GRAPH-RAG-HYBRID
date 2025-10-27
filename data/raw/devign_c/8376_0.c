BlockBackend *blk_new(uint64_t perm, uint64_t shared_perm)

{

    BlockBackend *blk;



    blk = g_new0(BlockBackend, 1);

    blk->refcnt = 1;

    blk->perm = perm;

    blk->shared_perm = shared_perm;

    blk_set_enable_write_cache(blk, true);



    qemu_co_mutex_init(&blk->public.throttle_group_member.throttled_reqs_lock);

    qemu_co_queue_init(&blk->public.throttle_group_member.throttled_reqs[0]);

    qemu_co_queue_init(&blk->public.throttle_group_member.throttled_reqs[1]);

    block_acct_init(&blk->stats);



    notifier_list_init(&blk->remove_bs_notifiers);

    notifier_list_init(&blk->insert_bs_notifiers);



    QTAILQ_INSERT_TAIL(&block_backends, blk, link);

    return blk;

}
