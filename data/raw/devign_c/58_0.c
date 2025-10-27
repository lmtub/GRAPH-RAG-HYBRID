static uint64_t timer_read(void *opaque, target_phys_addr_t addr, unsigned size)

{

    LM32TimerState *s = opaque;

    uint32_t r = 0;



    addr >>= 2;

    switch (addr) {

    case R_SR:

    case R_CR:

    case R_PERIOD:

        r = s->regs[addr];

        break;

    case R_SNAPSHOT:

        r = (uint32_t)ptimer_get_count(s->ptimer);

        break;

    default:

        error_report("lm32_timer: read access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }



    trace_lm32_timer_memory_read(addr << 2, r);

    return r;

}
