void qed_commit_l2_cache_entry(L2TableCache *l2_cache, CachedL2Table *l2_table)

{

    CachedL2Table *entry;



    entry = qed_find_l2_cache_entry(l2_cache, l2_table->offset);

    if (entry) {

        qed_unref_l2_cache_entry(entry);

        qed_unref_l2_cache_entry(l2_table);

        return;

    }



    if (l2_cache->n_entries >= MAX_L2_CACHE_SIZE) {

        entry = QTAILQ_FIRST(&l2_cache->entries);

        QTAILQ_REMOVE(&l2_cache->entries, entry, node);

        l2_cache->n_entries--;

        qed_unref_l2_cache_entry(entry);

    }



    l2_cache->n_entries++;

    QTAILQ_INSERT_TAIL(&l2_cache->entries, l2_table, node);

}
