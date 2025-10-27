static void blkdebug_debug_event(BlockDriverState *bs, BlkDebugEvent event)

{

    BDRVBlkdebugState *s = bs->opaque;

    struct BlkdebugRule *rule;

    bool injected;



    assert((int)event >= 0 && event < BLKDBG_EVENT_MAX);



    injected = false;

    s->new_state = s->state;

    QLIST_FOREACH(rule, &s->rules[event], next) {

        injected = process_rule(bs, rule, injected);

    }

    s->state = s->new_state;

}
