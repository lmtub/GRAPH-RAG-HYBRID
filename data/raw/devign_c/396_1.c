static bool bdrv_drain_recurse(BlockDriverState *bs, bool begin)

{

    BdrvChild *child, *tmp;

    bool waited;



    /* Ensure any pending metadata writes are submitted to bs->file.  */

    bdrv_drain_invoke(bs, begin);



    /* Wait for drained requests to finish */

    waited = BDRV_POLL_WHILE(bs, atomic_read(&bs->in_flight) > 0);



    QLIST_FOREACH_SAFE(child, &bs->children, next, tmp) {

        BlockDriverState *bs = child->bs;

        bool in_main_loop =

            qemu_get_current_aio_context() == qemu_get_aio_context();

        assert(bs->refcnt > 0);

        if (in_main_loop) {

            /* In case the recursive bdrv_drain_recurse processes a

             * block_job_defer_to_main_loop BH and modifies the graph,

             * let's hold a reference to bs until we are done.

             *

             * IOThread doesn't have such a BH, and it is not safe to call

             * bdrv_unref without BQL, so skip doing it there.

             */

            bdrv_ref(bs);

        }

        waited |= bdrv_drain_recurse(bs, begin);

        if (in_main_loop) {

            bdrv_unref(bs);

        }

    }



    return waited;

}
