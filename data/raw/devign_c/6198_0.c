void armv7m_nvic_set_pending(void *opaque, int irq, bool secure)

{

    NVICState *s = (NVICState *)opaque;

    bool banked = exc_is_banked(irq);

    VecInfo *vec;



    assert(irq > ARMV7M_EXCP_RESET && irq < s->num_irq);

    assert(!secure || banked);



    vec = (banked && secure) ? &s->sec_vectors[irq] : &s->vectors[irq];



    trace_nvic_set_pending(irq, secure, vec->enabled, vec->prio);



    if (irq >= ARMV7M_EXCP_HARD && irq < ARMV7M_EXCP_PENDSV) {

        /* If a synchronous exception is pending then it may be

         * escalated to HardFault if:

         *  * it is equal or lower priority to current execution

         *  * it is disabled

         * (ie we need to take it immediately but we can't do so).

         * Asynchronous exceptions (and interrupts) simply remain pending.

         *

         * For QEMU, we don't have any imprecise (asynchronous) faults,

         * so we can assume that PREFETCH_ABORT and DATA_ABORT are always

         * synchronous.

         * Debug exceptions are awkward because only Debug exceptions

         * resulting from the BKPT instruction should be escalated,

         * but we don't currently implement any Debug exceptions other

         * than those that result from BKPT, so we treat all debug exceptions

         * as needing escalation.

         *

         * This all means we can identify whether to escalate based only on

         * the exception number and don't (yet) need the caller to explicitly

         * tell us whether this exception is synchronous or not.

         */

        int running = nvic_exec_prio(s);

        bool escalate = false;



        if (vec->prio >= running) {

            trace_nvic_escalate_prio(irq, vec->prio, running);

            escalate = true;

        } else if (!vec->enabled) {

            trace_nvic_escalate_disabled(irq);

            escalate = true;

        }



        if (escalate) {

            if (running < 0) {

                /* We want to escalate to HardFault but we can't take a

                 * synchronous HardFault at this point either. This is a

                 * Lockup condition due to a guest bug. We don't model

                 * Lockup, so report via cpu_abort() instead.

                 */

                cpu_abort(&s->cpu->parent_obj,

                          "Lockup: can't escalate %d to HardFault "

                          "(current priority %d)\n", irq, running);

            }



            /* We can do the escalation, so we take HardFault instead.

             * If BFHFNMINS is set then we escalate to the banked HF for

             * the target security state of the original exception; otherwise

             * we take a Secure HardFault.

             */

            irq = ARMV7M_EXCP_HARD;

            if (arm_feature(&s->cpu->env, ARM_FEATURE_M_SECURITY) &&

                (secure ||

                 !(s->cpu->env.v7m.aircr & R_V7M_AIRCR_BFHFNMINS_MASK))) {

                vec = &s->sec_vectors[irq];

            } else {

                vec = &s->vectors[irq];

            }

            /* HF may be banked but there is only one shared HFSR */

            s->cpu->env.v7m.hfsr |= R_V7M_HFSR_FORCED_MASK;

        }

    }



    if (!vec->pending) {

        vec->pending = 1;

        nvic_irq_update(s);

    }

}
