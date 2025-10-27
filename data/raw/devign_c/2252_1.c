static void sh7750_mem_writel(void *opaque, target_phys_addr_t addr,

			      uint32_t mem_value)

{

    SH7750State *s = opaque;

    uint16_t temp;



    switch (addr) {

	/* SDRAM controller */

    case SH7750_BCR1_A7:

    case SH7750_BCR4_A7:

    case SH7750_WCR1_A7:

    case SH7750_WCR2_A7:

    case SH7750_WCR3_A7:

    case SH7750_MCR_A7:

	ignore_access("long write", addr);

	return;

	/* IO ports */

    case SH7750_PCTRA_A7:

	temp = porta_lines(s);

	s->pctra = mem_value;

	s->portdira = portdir(mem_value);

	s->portpullupa = portpullup(mem_value);

	porta_changed(s, temp);

	return;

    case SH7750_PCTRB_A7:

	temp = portb_lines(s);

	s->pctrb = mem_value;

	s->portdirb = portdir(mem_value);

	s->portpullupb = portpullup(mem_value);

	portb_changed(s, temp);

	return;

    case SH7750_MMUCR_A7:

	s->cpu->mmucr = mem_value;

	return;

    case SH7750_PTEH_A7:




	s->cpu->pteh = mem_value;

	return;

    case SH7750_PTEL_A7:

	s->cpu->ptel = mem_value;

	return;

    case SH7750_PTEA_A7:

	s->cpu->ptea = mem_value & 0x0000000f;

	return;

    case SH7750_TTB_A7:

	s->cpu->ttb = mem_value;

	return;

    case SH7750_TEA_A7:

	s->cpu->tea = mem_value;

	return;

    case SH7750_TRA_A7:

	s->cpu->tra = mem_value & 0x000007ff;

	return;

    case SH7750_EXPEVT_A7:

	s->cpu->expevt = mem_value & 0x000007ff;

	return;

    case SH7750_INTEVT_A7:

	s->cpu->intevt = mem_value & 0x000007ff;

	return;

    case SH7750_CCR_A7:

	s->ccr = mem_value;

	return;

    default:

	error_access("long write", addr);

	assert(0);

    }

}