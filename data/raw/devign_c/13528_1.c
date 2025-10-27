void qht_statistics_init(struct qht *ht, struct qht_stats *stats)

{

    struct qht_map *map;

    int i;



    map = atomic_rcu_read(&ht->map);



    stats->head_buckets = map->n_buckets;

    stats->used_head_buckets = 0;

    stats->entries = 0;

    qdist_init(&stats->chain);

    qdist_init(&stats->occupancy);



    for (i = 0; i < map->n_buckets; i++) {

        struct qht_bucket *head = &map->buckets[i];

        struct qht_bucket *b;

        unsigned int version;

        size_t buckets;

        size_t entries;

        int j;



        do {

            version = seqlock_read_begin(&head->sequence);

            buckets = 0;

            entries = 0;

            b = head;

            do {

                for (j = 0; j < QHT_BUCKET_ENTRIES; j++) {

                    if (atomic_read(&b->pointers[j]) == NULL) {

                        break;

                    }

                    entries++;

                }

                buckets++;

                b = atomic_rcu_read(&b->next);

            } while (b);

        } while (seqlock_read_retry(&head->sequence, version));



        if (entries) {

            qdist_inc(&stats->chain, buckets);

            qdist_inc(&stats->occupancy,

                      (double)entries / QHT_BUCKET_ENTRIES / buckets);

            stats->used_head_buckets++;

            stats->entries += entries;

        } else {

            qdist_inc(&stats->occupancy, 0);

        }

    }

}
