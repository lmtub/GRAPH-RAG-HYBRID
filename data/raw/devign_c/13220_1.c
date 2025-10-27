static void vmxnet3_set_events(VMXNET3State *s, uint32_t val)

{

    uint32_t events;



    VMW_CBPRN("Setting events: 0x%x", val);

    events = VMXNET3_READ_DRV_SHARED32(s->drv_shmem, ecr) | val;

    VMXNET3_WRITE_DRV_SHARED32(s->drv_shmem, ecr, events);

}
