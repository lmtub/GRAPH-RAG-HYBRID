void cpu_outw(pio_addr_t addr, uint16_t val)

{

    LOG_IOPORT("outw: %04"FMT_pioaddr" %04"PRIx16"\n", addr, val);

    trace_cpu_out(addr, val);

    ioport_write(1, addr, val);

}
