void bdrv_delete(BlockDriverState *bs)

{

    assert(!bs->peer);



    /* remove from list, if necessary */

    if (bs->device_name[0] != '\0') {

        QTAILQ_REMOVE(&bdrv_states, bs, list);

    }



    bdrv_close(bs);

    if (bs->file != NULL) {

        bdrv_delete(bs->file);

    }



    assert(bs != bs_snapshots);

    qemu_free(bs);

}
