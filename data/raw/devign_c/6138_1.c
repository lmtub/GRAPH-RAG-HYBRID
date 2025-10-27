static void qed_is_allocated_cb(void *opaque, int ret, uint64_t offset, size_t len)

{

    QEDIsAllocatedCB *cb = opaque;

    BDRVQEDState *s = cb->bs->opaque;

    *cb->pnum = len / BDRV_SECTOR_SIZE;

    switch (ret) {

    case QED_CLUSTER_FOUND:

        offset |= qed_offset_into_cluster(s, cb->pos);

        cb->status = BDRV_BLOCK_DATA | BDRV_BLOCK_OFFSET_VALID | offset;

        *cb->file = cb->bs->file->bs;

        break;

    case QED_CLUSTER_ZERO:

        cb->status = BDRV_BLOCK_ZERO;

        break;

    case QED_CLUSTER_L2:

    case QED_CLUSTER_L1:

        cb->status = 0;

        break;

    default:

        assert(ret < 0);

        cb->status = ret;

        break;

    }



    if (cb->co) {

        qemu_coroutine_enter(cb->co, NULL);

    }

}
