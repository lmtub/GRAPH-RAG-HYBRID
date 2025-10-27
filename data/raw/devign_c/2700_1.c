void bdrv_init(void)

{

    bdrv_register(&bdrv_raw);

    bdrv_register(&bdrv_host_device);


    bdrv_register(&bdrv_cow);


    bdrv_register(&bdrv_qcow);

    bdrv_register(&bdrv_vmdk);

    bdrv_register(&bdrv_cloop);

    bdrv_register(&bdrv_dmg);

    bdrv_register(&bdrv_bochs);

    bdrv_register(&bdrv_vpc);

    bdrv_register(&bdrv_vvfat);

    bdrv_register(&bdrv_qcow2);

    bdrv_register(&bdrv_parallels);


    bdrv_register(&bdrv_nbd);


}