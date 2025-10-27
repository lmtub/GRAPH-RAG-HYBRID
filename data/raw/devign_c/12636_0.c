void bdrv_replace_in_backing_chain(BlockDriverState *old, BlockDriverState *new)

{

    assert(!bdrv_requests_pending(old));

    assert(!bdrv_requests_pending(new));



    bdrv_ref(old);



    change_parent_backing_link(old, new);



    bdrv_unref(old);

}
