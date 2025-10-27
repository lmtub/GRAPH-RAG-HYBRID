static void coroutine_fn qed_aio_complete(QEDAIOCB *acb)

{

    BDRVQEDState *s = acb_to_s(acb);



    /* Free resources */

    qemu_iovec_destroy(&acb->cur_qiov);

    qed_unref_l2_cache_entry(acb->request.l2_table);



    /* Free the buffer we may have allocated for zero writes */

    if (acb->flags & QED_AIOCB_ZERO) {

        qemu_vfree(acb->qiov->iov[0].iov_base);

        acb->qiov->iov[0].iov_base = NULL;

    }



    /* Start next allocating write request waiting behind this one.  Note that

     * requests enqueue themselves when they first hit an unallocated cluster

     * but they wait until the entire request is finished before waking up the

     * next request in the queue.  This ensures that we don't cycle through

     * requests multiple times but rather finish one at a time completely.

     */

    if (acb == s->allocating_acb) {

        s->allocating_acb = NULL;

        if (!qemu_co_queue_empty(&s->allocating_write_reqs)) {

            qemu_co_enter_next(&s->allocating_write_reqs);

        } else if (s->header.features & QED_F_NEED_CHECK) {

            qed_start_need_check_timer(s);

        }

    }

}
