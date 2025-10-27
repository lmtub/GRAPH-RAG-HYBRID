void dma_acct_start(BlockDriverState *bs, BlockAcctCookie *cookie,

                    QEMUSGList *sg, enum BlockAcctType type)

{

    block_acct_start(bdrv_get_stats(bs), cookie, sg->size, type);

}
