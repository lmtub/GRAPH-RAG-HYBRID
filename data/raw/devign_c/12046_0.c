static uint64_t cadence_ttc_read(void *opaque, target_phys_addr_t offset,

    unsigned size)

{

    uint32_t ret = cadence_ttc_read_imp(opaque, offset);



    DB_PRINT("addr: %08x data: %08x\n", offset, ret);

    return ret;

}
