static void coroutine_fn bdrv_discard_co_entry(void *opaque)

{

    DiscardCo *rwco = opaque;



    rwco->ret = bdrv_co_discard(rwco->bs, rwco->sector_num, rwco->nb_sectors);

}
