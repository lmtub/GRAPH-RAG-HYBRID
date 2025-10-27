static void vmxnet3_update_pm_state(VMXNET3State *s)

{

    struct Vmxnet3_VariableLenConfDesc pm_descr;



    pm_descr.confLen =

        VMXNET3_READ_DRV_SHARED32(s->drv_shmem, devRead.pmConfDesc.confLen);

    pm_descr.confVer =

        VMXNET3_READ_DRV_SHARED32(s->drv_shmem, devRead.pmConfDesc.confVer);

    pm_descr.confPA =

        VMXNET3_READ_DRV_SHARED64(s->drv_shmem, devRead.pmConfDesc.confPA);



    vmxnet3_dump_conf_descr("PM State", &pm_descr);

}
