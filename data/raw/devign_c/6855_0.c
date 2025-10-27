static uint64_t uart_read(void *opaque, target_phys_addr_t addr,

                          unsigned size)

{

    LM32UartState *s = opaque;

    uint32_t r = 0;



    addr >>= 2;

    switch (addr) {

    case R_RXTX:

        r = s->regs[R_RXTX];

        s->regs[R_LSR] &= ~LSR_DR;

        uart_update_irq(s);

        break;

    case R_IIR:

    case R_LSR:

    case R_MSR:

        r = s->regs[addr];

        break;

    case R_IER:

    case R_LCR:

    case R_MCR:

    case R_DIV:

        error_report("lm32_uart: read access to write only register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    default:

        error_report("lm32_uart: read access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }



    trace_lm32_uart_memory_read(addr << 2, r);

    return r;

}
