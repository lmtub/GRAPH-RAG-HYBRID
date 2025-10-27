uint16_t cpu_inw(CPUState *env, pio_addr_t addr)

{

    uint16_t val;

    val = ioport_read(1, addr);

    LOG_IOPORT("inw : %04"FMT_pioaddr" %04"PRIx16"\n", addr, val);

#ifdef CONFIG_KQEMU

    if (env)

        env->last_io_time = cpu_get_time_fast();

#endif

    return val;

}
