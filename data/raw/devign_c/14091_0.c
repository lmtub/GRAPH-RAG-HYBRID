BlockReopenQueue *bdrv_reopen_queue(BlockReopenQueue *bs_queue,

                                    BlockDriverState *bs,

                                    QDict *options, int flags)

{

    assert(bs != NULL);



    BlockReopenQueueEntry *bs_entry;

    BdrvChild *child;

    QDict *old_options;



    if (bs_queue == NULL) {

        bs_queue = g_new0(BlockReopenQueue, 1);

        QSIMPLEQ_INIT(bs_queue);

    }



    if (!options) {

        options = qdict_new();

    }



    old_options = qdict_clone_shallow(bs->options);

    qdict_join(options, old_options, false);

    QDECREF(old_options);



    /* bdrv_open() masks this flag out */

    flags &= ~BDRV_O_PROTOCOL;



    QLIST_FOREACH(child, &bs->children, next) {

        int child_flags;



        if (child->bs->inherits_from != bs) {

            continue;

        }



        child_flags = child->role->inherit_flags(flags);

        /* TODO Pass down child flags (backing.*, extents.*, ...) */

        bdrv_reopen_queue(bs_queue, child->bs, NULL, child_flags);

    }



    bs_entry = g_new0(BlockReopenQueueEntry, 1);

    QSIMPLEQ_INSERT_TAIL(bs_queue, bs_entry, entry);



    bs_entry->state.bs = bs;

    bs_entry->state.options = options;

    bs_entry->state.flags = flags;



    return bs_queue;

}
