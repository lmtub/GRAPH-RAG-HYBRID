static int inject_error(BlockDriverState *bs, BlkdebugRule *rule)

{

    BDRVBlkdebugState *s = bs->opaque;

    int error = rule->options.inject.error;

    bool immediately = rule->options.inject.immediately;



    if (rule->options.inject.once) {

        QSIMPLEQ_REMOVE(&s->active_rules, rule, BlkdebugRule, active_next);

        remove_rule(rule);

    }



    if (!immediately) {

        aio_bh_schedule_oneshot(bdrv_get_aio_context(bs), error_callback_bh,

                                qemu_coroutine_self());

        qemu_coroutine_yield();

    }



    return -error;

}
