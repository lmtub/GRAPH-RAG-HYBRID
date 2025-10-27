mst_fpga_writeb(void *opaque, target_phys_addr_t addr, uint64_t value,

		unsigned size)

{

	mst_irq_state *s = (mst_irq_state *) opaque;

	value &= 0xffffffff;



	switch (addr) {

	case MST_LEDDAT1:

		s->leddat1 = value;

		break;

	case MST_LEDDAT2:

		s->leddat2 = value;

		break;

	case MST_LEDCTRL:

		s->ledctrl = value;

		break;

	case MST_GPSWR:

		s->gpswr = value;

		break;

	case MST_MSCWR1:

		s->mscwr1 = value;

		break;

	case MST_MSCWR2:

		s->mscwr2 = value;

		break;

	case MST_MSCWR3:

		s->mscwr3 = value;

		break;

	case MST_MSCRD:

		s->mscrd =  value;

		break;

	case MST_INTMSKENA:	/* Mask interrupt */

		s->intmskena = (value & 0xFEEFF);

		qemu_set_irq(s->parent, s->intsetclr & s->intmskena);

		break;

	case MST_INTSETCLR:	/* clear or set interrupt */

		s->intsetclr = (value & 0xFEEFF);

		qemu_set_irq(s->parent, s->intsetclr & s->intmskena);

		break;

		/* For PCMCIAx allow the to change only power and reset */

	case MST_PCMCIA0:

		s->pcmcia0 = (value & 0x1f) | (s->pcmcia0 & ~0x1f);

		break;

	case MST_PCMCIA1:

		s->pcmcia1 = (value & 0x1f) | (s->pcmcia1 & ~0x1f);

		break;

	default:

		printf("Mainstone - mst_fpga_writeb: Bad register offset "

			"0x" TARGET_FMT_plx "\n", addr);

	}

}
