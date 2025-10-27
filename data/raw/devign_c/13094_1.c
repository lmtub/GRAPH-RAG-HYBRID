static int blkdebug_debug_remove_breakpoint(BlockDriverState *bs,

                                            const char *tag)

{

    BDRVBlkdebugState *s = bs->opaque;

    BlkdebugSuspendedReq *r, *r_next;

    BlkdebugRule *rule, *next;

    int i, ret = -ENOENT;



    for (i = 0; i < BLKDBG__MAX; i++) {

        QLIST_FOREACH_SAFE(rule, &s->rules[i], next, next) {

            if (rule->action == ACTION_SUSPEND &&

                !strcmp(rule->options.suspend.tag, tag)) {

                remove_rule(rule);

                ret = 0;

            }

        }

    }

    QLIST_FOREACH_SAFE(r, &s->suspended_reqs, next, r_next) {

        if (!strcmp(r->tag, tag)) {

            qemu_coroutine_enter(r->co, NULL);

            ret = 0;

        }

    }

    return ret;

}
