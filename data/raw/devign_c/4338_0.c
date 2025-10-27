int qed_read_l2_table(BDRVQEDState *s, QEDRequest *request, uint64_t offset)

{

    int ret;



    qed_unref_l2_cache_entry(request->l2_table);



    /* Check for cached L2 entry */

    request->l2_table = qed_find_l2_cache_entry(&s->l2_cache, offset);

    if (request->l2_table) {

        return 0;

    }



    request->l2_table = qed_alloc_l2_cache_entry(&s->l2_cache);

    request->l2_table->table = qed_alloc_table(s);



    BLKDBG_EVENT(s->bs->file, BLKDBG_L2_LOAD);

    ret = qed_read_table(s, offset, request->l2_table->table);



    qed_acquire(s);

    if (ret) {

        /* can't trust loaded L2 table anymore */

        qed_unref_l2_cache_entry(request->l2_table);

        request->l2_table = NULL;

    } else {

        request->l2_table->offset = offset;



        qed_commit_l2_cache_entry(&s->l2_cache, request->l2_table);



        /* This is guaranteed to succeed because we just committed the entry

         * to the cache.

         */

        request->l2_table = qed_find_l2_cache_entry(&s->l2_cache, offset);

        assert(request->l2_table != NULL);

    }

    qed_release(s);



    return ret;

}
