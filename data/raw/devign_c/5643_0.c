assigned_dev_msix_mmio_read(void *opaque, target_phys_addr_t addr,

                            unsigned size)

{

    AssignedDevice *adev = opaque;

    uint64_t val;



    memcpy(&val, (void *)((uint8_t *)adev->msix_table + addr), size);



    return val;

}
