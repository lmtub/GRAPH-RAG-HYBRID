static bool bdrv_drain_recurse(BlockDriverState *bs)

{

    BdrvChild *child;

    bool waited;



    waited = BDRV_POLL_WHILE(bs, atomic_read(&bs->in_flight) > 0);



    if (bs->drv && bs->drv->bdrv_drain) {

        bs->drv->bdrv_drain(bs);

    }



    QLIST_FOREACH(child, &bs->children, next) {

        waited |= bdrv_drain_recurse(child->bs);

    }



    return waited;

}
