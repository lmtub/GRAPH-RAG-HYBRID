static uint64_t imx_timerg_read(void *opaque, target_phys_addr_t offset,

                                unsigned size)

{

    IMXTimerGState *s = (IMXTimerGState *)opaque;



    DPRINTF("g-read(offset=%x)", offset >> 2);

    switch (offset >> 2) {

    case 0: /* Control Register */

        DPRINTF(" cr = %x\n", s->cr);

        return s->cr;



    case 1: /* prescaler */

        DPRINTF(" pr = %x\n", s->pr);

        return s->pr;



    case 2: /* Status Register */

        DPRINTF(" sr = %x\n", s->sr);

        return s->sr;



    case 3: /* Interrupt Register */

        DPRINTF(" ir = %x\n", s->ir);

        return s->ir;



    case 4: /* Output Compare Register 1 */

        DPRINTF(" ocr1 = %x\n", s->ocr1);

        return s->ocr1;





    case 9: /* cnt */

        imx_timerg_update_counts(s);

        DPRINTF(" cnt = %x\n", s->cnt);

        return s->cnt;

    }



    IPRINTF("imx_timerg_read: Bad offset %x\n",

            (int)offset >> 2);

    return 0;

}
