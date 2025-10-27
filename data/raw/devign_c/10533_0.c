static void imx_gpt_compute_next_timeout(IMXGPTState *s, bool event)

{

    uint32_t timeout = TIMER_MAX;

    uint32_t count = 0;

    long long limit;



    if (!(s->cr & GPT_CR_EN)) {

        /* if not enabled just return */

        return;

    }



    if (event) {

        /* This is a timer event  */



        if ((s->cr & GPT_CR_FRR)  && (s->next_timeout != TIMER_MAX)) {

            /*

             * if we are in free running mode and we have not reached

             * the TIMER_MAX limit, then update the count

             */

            count = imx_gpt_update_count(s);

        }

    } else {

        /* not a timer event, then just update the count */



        count = imx_gpt_update_count(s);

    }



    /* now, find the next timeout related to count */



    if (s->ir & GPT_IR_OF1IE) {

        timeout = imx_gpt_find_limit(count, s->ocr1, timeout);

    }

    if (s->ir & GPT_IR_OF2IE) {

        timeout = imx_gpt_find_limit(count, s->ocr2, timeout);

    }

    if (s->ir & GPT_IR_OF3IE) {

        timeout = imx_gpt_find_limit(count, s->ocr3, timeout);

    }



    /* find the next set of interrupts to raise for next timer event */



    s->next_int = 0;

    if ((s->ir & GPT_IR_OF1IE) && (timeout == s->ocr1)) {

        s->next_int |= GPT_SR_OF1;

    }

    if ((s->ir & GPT_IR_OF2IE) && (timeout == s->ocr2)) {

        s->next_int |= GPT_SR_OF2;

    }

    if ((s->ir & GPT_IR_OF3IE) && (timeout == s->ocr3)) {

        s->next_int |= GPT_SR_OF3;

    }

    if ((s->ir & GPT_IR_ROVIE) && (timeout == TIMER_MAX)) {

        s->next_int |= GPT_SR_ROV;

    }



    /* the new range to count down from */

    limit = timeout - imx_gpt_update_count(s);



    if (limit < 0) {

        /*

         * if we reach here, then QEMU is running too slow and we pass the

         * timeout limit while computing it. Let's deliver the interrupt

         * and compute a new limit.

         */

        s->sr |= s->next_int;



        imx_gpt_compute_next_timeout(s, event);



        imx_gpt_update_int(s);

    } else {

        /* New timeout value */

        s->next_timeout = timeout;



        /* reset the limit to the computed range */

        ptimer_set_limit(s->timer, limit, 1);

    }

}
