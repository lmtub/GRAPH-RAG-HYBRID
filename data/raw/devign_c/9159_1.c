int monitor_read_bdrv_key_start(Monitor *mon, BlockDriverState *bs,

                                BlockCompletionFunc *completion_cb,

                                void *opaque)

{

    Error *local_err = NULL;

    int err;



    bdrv_add_key(bs, NULL, &local_err);

    if (!local_err) {

        if (completion_cb)

            completion_cb(opaque, 0);

        return 0;

    }



    /* Need a key for @bs */



    if (monitor_ctrl_mode(mon)) {

        qerror_report_err(local_err);


        return -1;

    }




    monitor_printf(mon, "%s (%s) is encrypted.\n", bdrv_get_device_name(bs),

                   bdrv_get_encrypted_filename(bs));



    mon->password_completion_cb = completion_cb;

    mon->password_opaque = opaque;



    err = monitor_read_password(mon, bdrv_password_cb, bs);



    if (err && completion_cb)

        completion_cb(opaque, err);



    return err;

}