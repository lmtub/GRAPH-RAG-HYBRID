static void update_sr (AC97LinkState *s, AC97BusMasterRegs *r, uint32_t new_sr)

{

    int event = 0;

    int level = 0;

    uint32_t new_mask = new_sr & SR_INT_MASK;

    uint32_t old_mask = r->sr & SR_INT_MASK;

    uint32_t masks[] = {GS_PIINT, GS_POINT, GS_MINT};



    if (new_mask ^ old_mask) {

        /** @todo is IRQ deasserted when only one of status bits is cleared? */

        if (!new_mask) {

            event = 1;

            level = 0;

        }

        else {

            if ((new_mask & SR_LVBCI) && (r->cr & CR_LVBIE)) {

                event = 1;

                level = 1;

            }

            if ((new_mask & SR_BCIS) && (r->cr & CR_IOCE)) {

                event = 1;

                level = 1;

            }

        }

    }



    r->sr = new_sr;



    dolog ("IOC%d LVB%d sr=%#x event=%d level=%d\n",

           r->sr & SR_BCIS, r->sr & SR_LVBCI,

           r->sr,

           event, level);



    if (!event)

        return;



    if (level) {

        s->glob_sta |= masks[r - s->bm_regs];

        dolog ("set irq level=1\n");

        qemu_set_irq (s->pci_dev->irq[0], 1);

    }

    else {

        s->glob_sta &= ~masks[r - s->bm_regs];

        dolog ("set irq level=0\n");

        qemu_set_irq (s->pci_dev->irq[0], 0);

    }

}
