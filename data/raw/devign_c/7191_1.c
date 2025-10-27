static bool qht_insert__locked(struct qht *ht, struct qht_map *map,

                               struct qht_bucket *head, void *p, uint32_t hash,

                               bool *needs_resize)

{

    struct qht_bucket *b = head;

    struct qht_bucket *prev = NULL;

    struct qht_bucket *new = NULL;

    int i;



    do {

        for (i = 0; i < QHT_BUCKET_ENTRIES; i++) {

            if (b->pointers[i]) {

                if (unlikely(b->pointers[i] == p)) {

                    return false;

                }

            } else {

                goto found;

            }

        }

        prev = b;

        b = b->next;

    } while (b);



    b = qemu_memalign(QHT_BUCKET_ALIGN, sizeof(*b));

    memset(b, 0, sizeof(*b));

    new = b;

    i = 0;

    atomic_inc(&map->n_added_buckets);

    if (unlikely(qht_map_needs_resize(map)) && needs_resize) {

        *needs_resize = true;

    }



 found:

    /* found an empty key: acquire the seqlock and write */

    seqlock_write_begin(&head->sequence);

    if (new) {

        atomic_rcu_set(&prev->next, b);

    }

    b->hashes[i] = hash;

    /* smp_wmb() implicit in seqlock_write_begin.  */

    atomic_set(&b->pointers[i], p);

    seqlock_write_end(&head->sequence);

    return true;

}
