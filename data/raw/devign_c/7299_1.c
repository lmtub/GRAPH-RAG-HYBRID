void blk_eject(BlockBackend *blk, bool eject_flag)

{

    BlockDriverState *bs = blk_bs(blk);

    char *id;



    /* blk_eject is only called by qdevified devices */

    assert(!blk->legacy_dev);



    if (bs) {

        bdrv_eject(bs, eject_flag);



        id = blk_get_attached_dev_id(blk);

        qapi_event_send_device_tray_moved(blk_name(blk), id,

                                          eject_flag, &error_abort);

        g_free(id);



    }

}
