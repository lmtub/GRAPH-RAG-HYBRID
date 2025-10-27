static void openpic_reset(DeviceState *d)

{

    OpenPICState *opp = FROM_SYSBUS(typeof (*opp), sysbus_from_qdev(d));

    int i;



    opp->glbc = GLBC_RESET;

    /* Initialise controller registers */

    opp->frep = ((opp->nb_irqs -1) << FREP_NIRQ_SHIFT) |

                ((opp->nb_cpus -1) << FREP_NCPU_SHIFT) |

                (opp->vid << FREP_VID_SHIFT);



    opp->pint = 0;

    opp->spve = -1 & opp->spve_mask;

    opp->tifr = opp->tifr_reset;

    /* Initialise IRQ sources */

    for (i = 0; i < opp->max_irq; i++) {

        opp->src[i].ipvp = opp->ipvp_reset;

        opp->src[i].ide  = opp->ide_reset;

    }

    /* Initialise IRQ destinations */

    for (i = 0; i < MAX_CPU; i++) {

        opp->dst[i].pctp      = 15;

        opp->dst[i].pcsr      = 0x00000000;

        memset(&opp->dst[i].raised, 0, sizeof(IRQ_queue_t));

        opp->dst[i].raised.next = -1;

        memset(&opp->dst[i].servicing, 0, sizeof(IRQ_queue_t));

        opp->dst[i].servicing.next = -1;

    }

    /* Initialise timers */

    for (i = 0; i < MAX_TMR; i++) {

        opp->timers[i].ticc = 0;

        opp->timers[i].tibc = TIBC_CI;

    }

    /* Go out of RESET state */

    opp->glbc = 0;

}
