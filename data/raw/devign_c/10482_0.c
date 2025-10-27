void blk_dev_change_media_cb(BlockBackend *blk, bool load)

{

    if (blk->dev_ops && blk->dev_ops->change_media_cb) {

        bool tray_was_closed = !blk_dev_is_tray_open(blk);



        blk->dev_ops->change_media_cb(blk->dev_opaque, load);

        if (tray_was_closed) {

            /* tray open */

            qapi_event_send_device_tray_moved(blk_name(blk),

                                              true, &error_abort);

        }

        if (load) {

            /* tray close */

            qapi_event_send_device_tray_moved(blk_name(blk),

                                              false, &error_abort);

        }

    }

}
