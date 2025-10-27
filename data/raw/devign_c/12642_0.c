static void pxa2xx_pm_write(void *opaque, target_phys_addr_t addr,

                uint32_t value)

{

    struct pxa2xx_state_s *s = (struct pxa2xx_state_s *) opaque;

    if (addr > s->pm_base + PCMD31) {

        /* Special case: PWRI2C registers appear in the same range.  */

        pxa2xx_i2c_write(s->i2c[1], addr, value);

        return;

    }

    addr -= s->pm_base;



    switch (addr) {

    case PMCR:

        s->pm_regs[addr >> 2] &= 0x15 & ~(value & 0x2a);

        s->pm_regs[addr >> 2] |= value & 0x15;

        break;



    case PSSR:	/* Read-clean registers */

    case RCSR:

    case PKSR:

        s->pm_regs[addr >> 2] &= ~value;

        break;



    default:	/* Read-write registers */

        if (addr >= PMCR && addr <= PCMD31 && !(addr & 3)) {

            s->pm_regs[addr >> 2] = value;

            break;

        }



        printf("%s: Bad register " REG_FMT "\n", __FUNCTION__, addr);

        break;

    }

}
