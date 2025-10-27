static void blk_delete(BlockBackend *blk)

{

    assert(!blk->refcnt);

    assert(!blk->name);

    assert(!blk->dev);




    if (blk->root) {

        blk_remove_bs(blk);


    assert(QLIST_EMPTY(&blk->remove_bs_notifiers.notifiers));

    assert(QLIST_EMPTY(&blk->insert_bs_notifiers.notifiers));

    QTAILQ_REMOVE(&block_backends, blk, link);

    drive_info_del(blk->legacy_dinfo);

    block_acct_cleanup(&blk->stats);

    g_free(blk);
