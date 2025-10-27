static int megasas_cache_flush(MegasasState *s, MegasasCmd *cmd)

{

    bdrv_drain_all();

    return MFI_STAT_OK;

}
