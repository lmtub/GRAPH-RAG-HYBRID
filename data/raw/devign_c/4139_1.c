static void quorum_aio_cb(void *opaque, int ret)

{

    QuorumChildRequest *sacb = opaque;

    QuorumAIOCB *acb = sacb->parent;

    BDRVQuorumState *s = acb->common.bs->opaque;



    sacb->ret = ret;

    acb->count++;

    if (ret == 0) {

        acb->success_count++;

    } else {

        quorum_report_bad(acb, sacb->aiocb->bs->node_name, ret);

    }

    assert(acb->count <= s->num_children);

    assert(acb->success_count <= s->num_children);

    if (acb->count < s->num_children) {

        return;

    }



    /* Do the vote on read */

    if (acb->is_read) {

        quorum_vote(acb);

    } else {

        quorum_has_too_much_io_failed(acb);

    }



    quorum_aio_finalize(acb);

}
