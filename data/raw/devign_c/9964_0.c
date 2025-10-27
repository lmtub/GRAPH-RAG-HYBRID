static void bdrv_rw_em_cb(void *opaque, int ret)

{

    *(int *)opaque = ret;

}
