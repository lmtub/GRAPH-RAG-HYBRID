static void virtio_blk_handle_read(VirtIOBlockReq *req)

{

    uint64_t sector;



    sector = virtio_ldq_p(VIRTIO_DEVICE(req->dev), &req->out.sector);



    bdrv_acct_start(req->dev->bs, &req->acct, req->qiov.size, BDRV_ACCT_READ);



    trace_virtio_blk_handle_read(req, sector, req->qiov.size / 512);



    if (sector & req->dev->sector_mask) {

        virtio_blk_rw_complete(req, -EIO);

        return;

    }

    if (req->qiov.size % req->dev->conf->logical_block_size) {

        virtio_blk_rw_complete(req, -EIO);

        return;

    }

    bdrv_aio_readv(req->dev->bs, sector, &req->qiov,

                   req->qiov.size / BDRV_SECTOR_SIZE,

                   virtio_blk_rw_complete, req);

}
