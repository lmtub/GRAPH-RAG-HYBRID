int blk_insert_bs(BlockBackend *blk, BlockDriverState *bs, Error **errp)

{

    blk->root = bdrv_root_attach_child(bs, "root", &child_root,

                                       blk->perm, blk->shared_perm, blk, errp);

    if (blk->root == NULL) {

        return -EPERM;

    }

    bdrv_ref(bs);



    notifier_list_notify(&blk->insert_bs_notifiers, blk);

    if (blk->public.throttle_group_member.throttle_state) {

        throttle_timers_attach_aio_context(

            &blk->public.throttle_group_member.throttle_timers,

            bdrv_get_aio_context(bs));

    }



    return 0;

}
