static void m5206_mbar_update(m5206_mbar_state *s)

{

    int irq;

    int vector;

    int level;



    irq = m5206_find_pending_irq(s);

    if (irq) {

        int tmp;

        tmp = s->icr[irq];

        level = (tmp >> 2) & 7;

        if (tmp & 0x80) {

            /* Autovector.  */

            vector = 24 + level;

        } else {

            switch (irq) {

            case 8: /* SWT */

                vector = s->swivr;

                break;

            case 12: /* UART1 */

                vector = s->uivr[0];

                break;

            case 13: /* UART2 */

                vector = s->uivr[1];

                break;

            default:

                /* Unknown vector.  */

                fprintf(stderr, "Unhandled vector for IRQ %d\n", irq);

                vector = 0xf;

                break;

            }

        }

    } else {

        level = 0;

        vector = 0;

    }

    m68k_set_irq_level(s->cpu, level, vector);

}
