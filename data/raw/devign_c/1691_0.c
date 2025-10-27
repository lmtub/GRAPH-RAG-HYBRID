static void timer_write(void *opaque, target_phys_addr_t addr,

                        uint64_t value, unsigned size)

{

    LM32TimerState *s = opaque;



    trace_lm32_timer_memory_write(addr, value);



    addr >>= 2;

    switch (addr) {

    case R_SR:

        s->regs[R_SR] &= ~SR_TO;

        break;

    case R_CR:

        s->regs[R_CR] = value;

        if (s->regs[R_CR] & CR_START) {

            ptimer_run(s->ptimer, 1);

        }

        if (s->regs[R_CR] & CR_STOP) {

            ptimer_stop(s->ptimer);

        }

        break;

    case R_PERIOD:

        s->regs[R_PERIOD] = value;

        ptimer_set_count(s->ptimer, value);

        break;

    case R_SNAPSHOT:

        error_report("lm32_timer: write access to read only register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    default:

        error_report("lm32_timer: write access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }

    timer_update_irq(s);

}
