static void update_refcount_discard(BlockDriverState *bs,

                                    uint64_t offset, uint64_t length)

{

    BDRVQcowState *s = bs->opaque;

    Qcow2DiscardRegion *d, *p, *next;



    QTAILQ_FOREACH(d, &s->discards, next) {

        uint64_t new_start = MIN(offset, d->offset);

        uint64_t new_end = MAX(offset + length, d->offset + d->bytes);



        if (new_end - new_start <= length + d->bytes) {

            /* There can't be any overlap, areas ending up here have no

             * references any more and therefore shouldn't get freed another

             * time. */

            assert(d->bytes + length == new_end - new_start);

            d->offset = new_start;

            d->bytes = new_end - new_start;

            goto found;

        }

    }



    d = g_malloc(sizeof(*d));

    *d = (Qcow2DiscardRegion) {

        .bs     = bs,

        .offset = offset,

        .bytes  = length,

    };

    QTAILQ_INSERT_TAIL(&s->discards, d, next);



found:

    /* Merge discard requests if they are adjacent now */

    QTAILQ_FOREACH_SAFE(p, &s->discards, next, next) {

        if (p == d

            || p->offset > d->offset + d->bytes

            || d->offset > p->offset + p->bytes)

        {

            continue;

        }



        /* Still no overlap possible */

        assert(p->offset == d->offset + d->bytes

            || d->offset == p->offset + p->bytes);



        QTAILQ_REMOVE(&s->discards, p, next);

        d->offset = MIN(d->offset, p->offset);

        d->bytes += p->bytes;


    }

}