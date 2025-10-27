static void qht_bucket_reset__locked(struct qht_bucket *head)

{

    struct qht_bucket *b = head;

    int i;



    seqlock_write_begin(&head->sequence);

    do {

        for (i = 0; i < QHT_BUCKET_ENTRIES; i++) {

            if (b->pointers[i] == NULL) {

                goto done;

            }

            b->hashes[i] = 0;

            atomic_set(&b->pointers[i], NULL);

        }

        b = b->next;

    } while (b);

 done:

    seqlock_write_end(&head->sequence);

}
