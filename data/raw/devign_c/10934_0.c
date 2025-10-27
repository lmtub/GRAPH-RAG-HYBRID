static void check_exception(PowerPCCPU *cpu, sPAPRMachineState *spapr,

                            uint32_t token, uint32_t nargs,

                            target_ulong args,

                            uint32_t nret, target_ulong rets)

{

    uint32_t mask, buf, len, event_len;

    uint64_t xinfo;

    sPAPREventLogEntry *event;

    struct rtas_error_log *hdr;



    if ((nargs < 6) || (nargs > 7) || nret != 1) {

        rtas_st(rets, 0, RTAS_OUT_PARAM_ERROR);

        return;

    }



    xinfo = rtas_ld(args, 1);

    mask = rtas_ld(args, 2);

    buf = rtas_ld(args, 4);

    len = rtas_ld(args, 5);

    if (nargs == 7) {

        xinfo |= (uint64_t)rtas_ld(args, 6) << 32;

    }



    event = rtas_event_log_dequeue(mask, true);

    if (!event) {

        goto out_no_events;

    }



    hdr = event->data;

    event_len = be32_to_cpu(hdr->extended_length) + sizeof(*hdr);



    if (event_len < len) {

        len = event_len;

    }



    cpu_physical_memory_write(buf, event->data, len);

    rtas_st(rets, 0, RTAS_OUT_SUCCESS);

    g_free(event->data);

    g_free(event);



    /* according to PAPR+, the IRQ must be left asserted, or re-asserted, if

     * there are still pending events to be fetched via check-exception. We

     * do the latter here, since our code relies on edge-triggered

     * interrupts.

     */

    if (rtas_event_log_contains(mask, true)) {

        qemu_irq_pulse(xics_get_qirq(spapr->xics, spapr->check_exception_irq));

    }



    return;



out_no_events:

    rtas_st(rets, 0, RTAS_OUT_NO_ERRORS_FOUND);

}
