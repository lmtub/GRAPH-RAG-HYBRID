static void virtio_blk_handle_write(BlockRequest *blkreq, int *num_writes,

    VirtIOBlockReq *req)

{

    if (req->out->sector & req->dev->sector_mask) {

        virtio_blk_rw_complete(req, -EIO);

        return;

    }



    if (*num_writes == 32) {

        do_multiwrite(req->dev->bs, blkreq, *num_writes);

        *num_writes = 0;

    }



    blkreq[*num_writes].sector = req->out->sector;

    blkreq[*num_writes].nb_sectors = req->qiov.size / BDRV_SECTOR_SIZE;

    blkreq[*num_writes].qiov = &req->qiov;

    blkreq[*num_writes].cb = virtio_blk_rw_complete;

    blkreq[*num_writes].opaque = req;

    blkreq[*num_writes].error = 0;



    (*num_writes)++;

}
