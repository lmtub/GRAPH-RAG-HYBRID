static void replication_close(BlockDriverState *bs)

{

    BDRVReplicationState *s = bs->opaque;



    if (s->replication_state == BLOCK_REPLICATION_RUNNING) {

        replication_stop(s->rs, false, NULL);







    if (s->mode == REPLICATION_MODE_SECONDARY) {

        g_free(s->top_id);




    replication_remove(s->rs);
