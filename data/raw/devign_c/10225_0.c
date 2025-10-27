void bdrv_detach(BlockDriverState *bs, DeviceState *qdev)

{

    assert(bs->peer == qdev);

    bs->peer = NULL;

    bs->change_cb = NULL;

    bs->change_opaque = NULL;

}
