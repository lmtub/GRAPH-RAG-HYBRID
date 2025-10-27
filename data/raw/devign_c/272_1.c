static MemTxResult vtd_mem_ir_write(void *opaque, hwaddr addr,

                                    uint64_t value, unsigned size,

                                    MemTxAttrs attrs)

{

    int ret = 0;

    MSIMessage from = {0}, to = {0};



    from.address = (uint64_t) addr + VTD_INTERRUPT_ADDR_FIRST;

    from.data = (uint32_t) value;



    ret = vtd_interrupt_remap_msi(opaque, &from, &to);

    if (ret) {

        /* TODO: report error */

        VTD_DPRINTF(GENERAL, "int remap fail for addr 0x%"PRIx64

                    " data 0x%"PRIx32, from.address, from.data);

        /* Drop this interrupt */

        return MEMTX_ERROR;

    }



    VTD_DPRINTF(IR, "delivering MSI 0x%"PRIx64":0x%"PRIx32

                " for device sid 0x%04x",

                to.address, to.data, sid);



    if (dma_memory_write(&address_space_memory, to.address,

                         &to.data, size)) {

        VTD_DPRINTF(GENERAL, "error: fail to write 0x%"PRIx64

                    " value 0x%"PRIx32, to.address, to.data);

    }



    return MEMTX_OK;

}
