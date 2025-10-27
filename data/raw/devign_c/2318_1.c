static BlockAIOCB *blkverify_aio_readv(BlockDriverState *bs,

        int64_t sector_num, QEMUIOVector *qiov, int nb_sectors,

        BlockCompletionFunc *cb, void *opaque)

{

    BDRVBlkverifyState *s = bs->opaque;

    BlkverifyAIOCB *acb = blkverify_aio_get(bs, false, sector_num, qiov,

                                            nb_sectors, cb, opaque);



    acb->verify = blkverify_verify_readv;

    acb->buf = qemu_blockalign(bs->file->bs, qiov->size);

    qemu_iovec_init(&acb->raw_qiov, acb->qiov->niov);

    qemu_iovec_clone(&acb->raw_qiov, qiov, acb->buf);



    bdrv_aio_readv(s->test_file, sector_num, qiov, nb_sectors,

                   blkverify_aio_cb, acb);

    bdrv_aio_readv(bs->file, sector_num, &acb->raw_qiov, nb_sectors,

                   blkverify_aio_cb, acb);

    return &acb->common;

}
