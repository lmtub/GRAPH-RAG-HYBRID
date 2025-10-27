bdrv_acct_start(BlockDriverState *bs, BlockAcctCookie *cookie, int64_t bytes,

        enum BlockAcctType type)

{

    assert(type < BDRV_MAX_IOTYPE);



    cookie->bytes = bytes;

    cookie->start_time_ns = get_clock();

    cookie->type = type;

}
