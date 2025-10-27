static BlockAIOCB *read_fifo_child(QuorumAIOCB *acb)

{

    BDRVQuorumState *s = acb->common.bs->opaque;



    acb->qcrs[acb->child_iter].buf =

        qemu_blockalign(s->children[acb->child_iter]->bs, acb->qiov->size);

    qemu_iovec_init(&acb->qcrs[acb->child_iter].qiov, acb->qiov->niov);

    qemu_iovec_clone(&acb->qcrs[acb->child_iter].qiov, acb->qiov,

                     acb->qcrs[acb->child_iter].buf);

    bdrv_aio_readv(s->children[acb->child_iter]->bs, acb->sector_num,

                   &acb->qcrs[acb->child_iter].qiov, acb->nb_sectors,

                   quorum_aio_cb, &acb->qcrs[acb->child_iter]);



    return &acb->common;

}
