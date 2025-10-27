void bdrv_io_unplugged_end(BlockDriverState *bs)

{

    BdrvChild *child;



    assert(bs->io_plug_disabled);

    QLIST_FOREACH(child, &bs->children, next) {

        bdrv_io_unplugged_end(child->bs);

    }



    if (--bs->io_plug_disabled == 0 && bs->io_plugged > 0) {

        BlockDriver *drv = bs->drv;

        if (drv && drv->bdrv_io_plug) {

            drv->bdrv_io_plug(bs);

        }

    }

}
