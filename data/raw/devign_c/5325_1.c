static void quorum_aio_finalize(QuorumAIOCB *acb)

{

    BDRVQuorumState *s = acb->common.bs->opaque;

    int i, ret = 0;



    if (acb->vote_ret) {

        ret = acb->vote_ret;

    }



    acb->common.cb(acb->common.opaque, ret);



    if (acb->is_read) {

        for (i = 0; i < s->num_children; i++) {

            qemu_vfree(acb->qcrs[i].buf);

            qemu_iovec_destroy(&acb->qcrs[i].qiov);

        }

    }



    g_free(acb->qcrs);

    qemu_aio_release(acb);

}
