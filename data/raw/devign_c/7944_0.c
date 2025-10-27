static void platform_mmio_write(ReadWriteHandler *handler, pcibus_t addr,

                                uint32_t val, int len)

{

    DPRINTF("Warning: attempted write of 0x%x to physical "

            "address 0x" TARGET_FMT_plx " in xen platform mmio space\n",

            val, addr);

}
