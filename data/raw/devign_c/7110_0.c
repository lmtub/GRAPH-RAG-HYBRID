uint8_t cpu_inb(pio_addr_t addr)

{

    uint8_t val;

    val = ioport_read(0, addr);

    trace_cpu_in(addr, val);

    LOG_IOPORT("inb : %04"FMT_pioaddr" %02"PRIx8"\n", addr, val);

    return val;

}
