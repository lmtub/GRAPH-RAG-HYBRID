static void cs_mem_write(void *opaque, target_phys_addr_t addr,

                         uint64_t val, unsigned size)

{

    CSState *s = opaque;

    uint32_t saddr;



    saddr = addr >> 2;

    trace_cs4231_mem_writel_reg(saddr, s->regs[saddr], val);

    switch (saddr) {

    case 1:

        trace_cs4231_mem_writel_dreg(CS_RAP(s), s->dregs[CS_RAP(s)], val);

        switch(CS_RAP(s)) {

        case 11:

        case 25: // Read only

            break;

        case 12:

            val &= 0x40;

            val |= CS_CDC_VER; // Codec version

            s->dregs[CS_RAP(s)] = val;

            break;

        default:

            s->dregs[CS_RAP(s)] = val;

            break;

        }

        break;

    case 2: // Read only

        break;

    case 4:

        if (val & 1) {

            cs_reset(&s->busdev.qdev);

        }

        val &= 0x7f;

        s->regs[saddr] = val;

        break;

    default:

        s->regs[saddr] = val;

        break;

    }

}
