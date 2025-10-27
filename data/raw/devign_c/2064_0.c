static void ac97_write(void *opaque, target_phys_addr_t addr, uint64_t value,

                       unsigned size)

{

    MilkymistAC97State *s = opaque;



    trace_milkymist_ac97_memory_write(addr, value);



    addr >>= 2;

    switch (addr) {

    case R_AC97_CTRL:

        /* always raise an IRQ according to the direction */

        if (value & AC97_CTRL_RQEN) {

            if (value & AC97_CTRL_WRITE) {

                trace_milkymist_ac97_pulse_irq_crrequest();

                qemu_irq_pulse(s->crrequest_irq);

            } else {

                trace_milkymist_ac97_pulse_irq_crreply();

                qemu_irq_pulse(s->crreply_irq);

            }

        }



        /* RQEN is self clearing */

        s->regs[addr] = value & ~AC97_CTRL_RQEN;

        break;

    case R_D_CTRL:

    case R_U_CTRL:

        s->regs[addr] = value;

        update_voices(s);

        break;

    case R_AC97_ADDR:

    case R_AC97_DATAOUT:

    case R_AC97_DATAIN:

    case R_D_ADDR:

    case R_D_REMAINING:

    case R_U_ADDR:

    case R_U_REMAINING:

        s->regs[addr] = value;

        break;



    default:

        error_report("milkymist_ac97: write access to unknown register 0x"

                TARGET_FMT_plx, addr);

        break;

    }



}
