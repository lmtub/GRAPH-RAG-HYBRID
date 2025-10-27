static void qed_commit_l2_update(void *opaque, int ret)

{

    QEDAIOCB *acb = opaque;

    BDRVQEDState *s = acb_to_s(acb);

    CachedL2Table *l2_table = acb->request.l2_table;



    qed_commit_l2_cache_entry(&s->l2_cache, l2_table);



    /* This is guaranteed to succeed because we just committed the entry to the

     * cache.

     */

    acb->request.l2_table = qed_find_l2_cache_entry(&s->l2_cache,

                                                    l2_table->offset);

    assert(acb->request.l2_table != NULL);



    qed_aio_next_io(opaque, ret);

}
