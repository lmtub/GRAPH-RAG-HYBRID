void virtio_blk_data_plane_create(VirtIODevice *vdev, VirtIOBlkConf *blk,

                                  VirtIOBlockDataPlane **dataplane,

                                  Error **errp)

{

    VirtIOBlockDataPlane *s;

    Error *local_err = NULL;



    *dataplane = NULL;



    if (!blk->data_plane) {

        return;

    }



    if (blk->scsi) {

        error_setg(errp,

                   "device is incompatible with x-data-plane, use scsi=off");

        return;

    }



    if (blk->config_wce) {

        error_setg(errp, "device is incompatible with x-data-plane, "

                         "use config-wce=off");

        return;

    }



    /* If dataplane is (re-)enabled while the guest is running there could be

     * block jobs that can conflict.

     */

    if (bdrv_op_is_blocked(blk->conf.bs, BLOCK_OP_TYPE_DATAPLANE, &local_err)) {

        error_report("cannot start dataplane thread: %s",

                      error_get_pretty(local_err));

        error_free(local_err);

        return;

    }



    s = g_new0(VirtIOBlockDataPlane, 1);

    s->vdev = vdev;

    s->blk = blk;



    if (blk->iothread) {

        s->iothread = blk->iothread;

        object_ref(OBJECT(s->iothread));

    } else {

        /* Create per-device IOThread if none specified.  This is for

         * x-data-plane option compatibility.  If x-data-plane is removed we

         * can drop this.

         */

        object_initialize(&s->internal_iothread_obj,

                          sizeof(s->internal_iothread_obj),

                          TYPE_IOTHREAD);

        user_creatable_complete(OBJECT(&s->internal_iothread_obj), &error_abort);

        s->iothread = &s->internal_iothread_obj;

    }

    s->ctx = iothread_get_aio_context(s->iothread);



    error_setg(&s->blocker, "block device is in use by data plane");

    bdrv_op_block_all(blk->conf.bs, s->blocker);



    *dataplane = s;

}
