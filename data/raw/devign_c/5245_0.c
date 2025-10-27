static void uart_write(void *opaque, target_phys_addr_t addr, uint64_t value,

                       unsigned size)

{

    MilkymistUartState *s = opaque;

    unsigned char ch = value;



    trace_milkymist_uart_memory_write(addr, value);



    addr >>= 2;

    switch (addr) {

    case R_RXTX:

        if (s->chr) {

            qemu_chr_fe_write(s->chr, &ch, 1);

        }

        s->regs[R_STAT] |= STAT_TX_EVT;

        break;

    case R_DIV:

    case R_CTRL:

    case R_DBG:

        s->regs[addr] = value;

        break;



    case R_STAT:

        /* write one to clear bits */

        s->regs[addr] &= ~(value & (STAT_RX_EVT | STAT_TX_EVT));

        break;



    default:

        error_report("milkymist_uart: write access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }



    uart_update_irq(s);

}
