static void omap_tipb_bridge_write(void *opaque, target_phys_addr_t addr,

                                   uint64_t value, unsigned size)

{

    struct omap_tipb_bridge_s *s = (struct omap_tipb_bridge_s *) opaque;



    if (size < 2) {

        return omap_badwidth_write16(opaque, addr, value);

    }



    switch (addr) {

    case 0x00:	/* TIPB_CNTL */

        s->control = value & 0xffff;

        break;



    case 0x04:	/* TIPB_BUS_ALLOC */

        s->alloc = value & 0x003f;

        break;



    case 0x08:	/* MPU_TIPB_CNTL */

        s->buffer = value & 0x0003;

        break;



    case 0x0c:	/* ENHANCED_TIPB_CNTL */

        s->width_intr = !(value & 2);

        s->enh_control = value & 0x000f;

        break;



    case 0x10:	/* ADDRESS_DBG */

    case 0x14:	/* DATA_DEBUG_LOW */

    case 0x18:	/* DATA_DEBUG_HIGH */

    case 0x1c:	/* DEBUG_CNTR_SIG */

        OMAP_RO_REG(addr);

        break;



    default:

        OMAP_BAD_REG(addr);

    }

}
