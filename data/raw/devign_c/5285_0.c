static void suspend_request(BlockDriverState *bs, BlkdebugRule *rule)

{

    BDRVBlkdebugState *s = bs->opaque;

    BlkdebugSuspendedReq r;



    r = (BlkdebugSuspendedReq) {

        .co         = qemu_coroutine_self(),

        .tag        = g_strdup(rule->options.suspend.tag),

    };



    remove_rule(rule);

    QLIST_INSERT_HEAD(&s->suspended_reqs, &r, next);



    printf("blkdebug: Suspended request '%s'\n", r.tag);

    qemu_coroutine_yield();

    printf("blkdebug: Resuming request '%s'\n", r.tag);



    QLIST_REMOVE(&r, next);

    g_free(r.tag);

}
