mst_fpga_readb(void *opaque, target_phys_addr_t addr)

{

	mst_irq_state *s = (mst_irq_state *) opaque;



	switch (addr) {

	case MST_LEDDAT1:

		return s->leddat1;

	case MST_LEDDAT2:

		return s->leddat2;

	case MST_LEDCTRL:

		return s->ledctrl;

	case MST_GPSWR:

		return s->gpswr;

	case MST_MSCWR1:

		return s->mscwr1;

	case MST_MSCWR2:

		return s->mscwr2;

	case MST_MSCWR3:

		return s->mscwr3;

	case MST_MSCRD:

		return s->mscrd;

	case MST_INTMSKENA:

		return s->intmskena;

	case MST_INTSETCLR:

		return s->intsetclr;

	case MST_PCMCIA0:

		return s->pcmcia0;

	case MST_PCMCIA1:

		return s->pcmcia1;

	default:

		printf("Mainstone - mst_fpga_readb: Bad register offset "

			"0x" TARGET_FMT_plx " \n", addr);

	}

	return 0;

}
