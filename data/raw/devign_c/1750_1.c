static void vm_completion(ReadLineState *rs, const char *str)

{

    size_t len;

    BlockDriverState *bs;

    BdrvNextIterator *it = NULL;



    len = strlen(str);

    readline_set_completion_index(rs, len);



    while ((it = bdrv_next(it, &bs))) {

        SnapshotInfoList *snapshots, *snapshot;

        AioContext *ctx = bdrv_get_aio_context(bs);

        bool ok = false;



        aio_context_acquire(ctx);

        if (bdrv_can_snapshot(bs)) {

            ok = bdrv_query_snapshot_info_list(bs, &snapshots, NULL) == 0;

        }

        aio_context_release(ctx);

        if (!ok) {

            continue;

        }



        snapshot = snapshots;

        while (snapshot) {

            char *completion = snapshot->value->name;

            if (!strncmp(str, completion, len)) {

                readline_add_completion(rs, completion);

            }

            completion = snapshot->value->id;

            if (!strncmp(str, completion, len)) {

                readline_add_completion(rs, completion);

            }

            snapshot = snapshot->next;

        }

        qapi_free_SnapshotInfoList(snapshots);

    }



}
