uint8_t cpu_inb(CPUState *env, pio_addr_t addr)

{

    uint8_t val;

    val = ioport_read(0, addr);

    LOG_IOPORT("inb : %04"FMT_pioaddr" %02"PRIx8"\n", addr, val);

#ifdef CONFIG_KQEMU

    if (env)

        env->last_io_time = cpu_get_time_fast();

#endif

    return val;

}
