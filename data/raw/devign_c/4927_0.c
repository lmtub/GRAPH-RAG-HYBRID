void cpu_outl(pio_addr_t addr, uint32_t val)

{

    LOG_IOPORT("outl: %04"FMT_pioaddr" %08"PRIx32"\n", addr, val);

    trace_cpu_out(addr, val);

    ioport_write(2, addr, val);

}
