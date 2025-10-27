static uint64_t gem_read(void *opaque, target_phys_addr_t offset, unsigned size)

{

    GemState *s;

    uint32_t retval;



    s = (GemState *)opaque;



    offset >>= 2;

    retval = s->regs[offset];



    DB_PRINT("offset: 0x%04x read: 0x%08x\n", offset*4, retval);



    switch (offset) {

    case GEM_ISR:

        qemu_set_irq(s->irq, 0);

        break;

    case GEM_PHYMNTNC:

        if (retval & GEM_PHYMNTNC_OP_R) {

            uint32_t phy_addr, reg_num;



            phy_addr = (retval & GEM_PHYMNTNC_ADDR) >> GEM_PHYMNTNC_ADDR_SHFT;

            if (phy_addr == BOARD_PHY_ADDRESS) {

                reg_num = (retval & GEM_PHYMNTNC_REG) >> GEM_PHYMNTNC_REG_SHIFT;

                retval &= 0xFFFF0000;

                retval |= gem_phy_read(s, reg_num);

            } else {

                retval |= 0xFFFF; /* No device at this address */

            }

        }

        break;

    }



    /* Squash read to clear bits */

    s->regs[offset] &= ~(s->regs_rtc[offset]);



    /* Do not provide write only bits */

    retval &= ~(s->regs_wo[offset]);



    DB_PRINT("0x%08x\n", retval);

    return retval;

}
