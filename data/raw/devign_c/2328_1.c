static void quorum_vote(QuorumAIOCB *acb)

{

    bool quorum = true;

    int i, j, ret;

    QuorumVoteValue hash;

    BDRVQuorumState *s = acb->common.bs->opaque;

    QuorumVoteVersion *winner;



    if (quorum_has_too_much_io_failed(acb)) {

        return;

    }



    /* get the index of the first successful read */

    for (i = 0; i < s->num_children; i++) {

        if (!acb->qcrs[i].ret) {

            break;

        }

    }



    assert(i < s->num_children);



    /* compare this read with all other successful reads stopping at quorum

     * failure

     */

    for (j = i + 1; j < s->num_children; j++) {

        if (acb->qcrs[j].ret) {

            continue;

        }

        quorum = quorum_compare(acb, &acb->qcrs[i].qiov, &acb->qcrs[j].qiov);

        if (!quorum) {

            break;

       }

    }



    /* Every successful read agrees */

    if (quorum) {

        quorum_copy_qiov(acb->qiov, &acb->qcrs[i].qiov);

        return;

    }



    /* compute hashes for each successful read, also store indexes */

    for (i = 0; i < s->num_children; i++) {

        if (acb->qcrs[i].ret) {

            continue;

        }

        ret = quorum_compute_hash(acb, i, &hash);

        /* if ever the hash computation failed */

        if (ret < 0) {

            acb->vote_ret = ret;

            goto free_exit;

        }

        quorum_count_vote(&acb->votes, &hash, i);

    }



    /* vote to select the most represented version */

    winner = quorum_get_vote_winner(&acb->votes);



    /* if the winner count is smaller than threshold the read fails */

    if (winner->vote_count < s->threshold) {

        quorum_report_failure(acb);

        acb->vote_ret = -EIO;

        goto free_exit;

    }



    /* we have a winner: copy it */

    quorum_copy_qiov(acb->qiov, &acb->qcrs[winner->index].qiov);



    /* some versions are bad print them */

    quorum_report_bad_versions(s, acb, &winner->value);



free_exit:

    /* free lists */

    quorum_free_vote_list(&acb->votes);

}
