static void memcard_write(void *opaque, target_phys_addr_t addr, uint64_t value,

                          unsigned size)

{

    MilkymistMemcardState *s = opaque;



    trace_milkymist_memcard_memory_write(addr, value);



    addr >>= 2;

    switch (addr) {

    case R_PENDING:

        /* clear rx pending bits */

        s->regs[R_PENDING] &= ~(value & (PENDING_CMD_RX | PENDING_DAT_RX));

        update_pending_bits(s);

        break;

    case R_CMD:

        if (!s->enabled) {

            break;

        }

        if (s->ignore_next_cmd) {

            s->ignore_next_cmd = 0;

            break;

        }

        s->command[s->command_write_ptr] = value & 0xff;

        s->command_write_ptr = (s->command_write_ptr + 1) % 6;

        if (s->command_write_ptr == 0) {

            memcard_sd_command(s);

        }

        break;

    case R_DAT:

        if (!s->enabled) {

            break;

        }

        sd_write_data(s->card, (value >> 24) & 0xff);

        sd_write_data(s->card, (value >> 16) & 0xff);

        sd_write_data(s->card, (value >> 8) & 0xff);

        sd_write_data(s->card, value & 0xff);

        break;

    case R_ENABLE:

        s->regs[addr] = value;

        update_pending_bits(s);

        break;

    case R_CLK2XDIV:

    case R_START:

        s->regs[addr] = value;

        break;



    default:

        error_report("milkymist_memcard: write access to unknown register 0x"

                TARGET_FMT_plx, addr << 2);

        break;

    }

}
