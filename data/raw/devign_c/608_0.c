static void ahci_mem_write(void *opaque, target_phys_addr_t addr,

                           uint64_t val, unsigned size)

{

    AHCIState *s = opaque;



    /* Only aligned reads are allowed on AHCI */

    if (addr & 3) {

        fprintf(stderr, "ahci: Mis-aligned write to addr 0x"

                TARGET_FMT_plx "\n", addr);

        return;

    }



    if (addr < AHCI_GENERIC_HOST_CONTROL_REGS_MAX_ADDR) {

        DPRINTF(-1, "(addr 0x%08X), val 0x%08"PRIX64"\n", (unsigned) addr, val);



        switch (addr) {

            case HOST_CAP: /* R/WO, RO */

                /* FIXME handle R/WO */

                break;

            case HOST_CTL: /* R/W */

                if (val & HOST_CTL_RESET) {

                    DPRINTF(-1, "HBA Reset\n");

                    ahci_reset(s);

                } else {

                    s->control_regs.ghc = (val & 0x3) | HOST_CTL_AHCI_EN;

                    ahci_check_irq(s);

                }

                break;

            case HOST_IRQ_STAT: /* R/WC, RO */

                s->control_regs.irqstatus &= ~val;

                ahci_check_irq(s);

                break;

            case HOST_PORTS_IMPL: /* R/WO, RO */

                /* FIXME handle R/WO */

                break;

            case HOST_VERSION: /* RO */

                /* FIXME report write? */

                break;

            default:

                DPRINTF(-1, "write to unknown register 0x%x\n", (unsigned)addr);

        }

    } else if ((addr >= AHCI_PORT_REGS_START_ADDR) &&

               (addr < (AHCI_PORT_REGS_START_ADDR +

                (s->ports * AHCI_PORT_ADDR_OFFSET_LEN)))) {

        ahci_port_write(s, (addr - AHCI_PORT_REGS_START_ADDR) >> 7,

                        addr & AHCI_PORT_ADDR_OFFSET_MASK, val);

    }



}
