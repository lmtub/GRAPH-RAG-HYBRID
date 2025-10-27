blkdebug_co_preadv(BlockDriverState *bs, uint64_t offset, uint64_t bytes,
                   QEMUIOVector *qiov, int flags)
{
    BDRVBlkdebugState *s = bs->opaque;
    BlkdebugRule *rule = NULL;
    QSIMPLEQ_FOREACH(rule, &s->active_rules, active_next) {
        uint64_t inject_offset = rule->options.inject.offset;
        if (inject_offset == -1 ||
            (inject_offset >= offset && inject_offset < offset + bytes))
        {
            break;
    if (rule && rule->options.inject.error) {
        return inject_error(bs, rule);
    return bdrv_co_preadv(bs->file, offset, bytes, qiov, flags);