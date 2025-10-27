void bdrv_init_with_whitelist(void)

{

    use_bdrv_whitelist = 1;

    bdrv_init();

}
