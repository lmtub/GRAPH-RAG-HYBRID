static void vmxnet3_update_rx_mode(VMXNET3State *s)

{

    s->rx_mode = VMXNET3_READ_DRV_SHARED32(s->drv_shmem,

                                           devRead.rxFilterConf.rxMode);

    VMW_CFPRN("RX mode: 0x%08X", s->rx_mode);

}
