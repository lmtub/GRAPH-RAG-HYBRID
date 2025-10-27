static bool vmxnet3_verify_driver_magic(hwaddr dshmem)

{

    return (VMXNET3_READ_DRV_SHARED32(dshmem, magic) == VMXNET3_REV1_MAGIC);

}
