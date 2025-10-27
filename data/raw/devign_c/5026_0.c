void spapr_events_init(sPAPREnvironment *spapr)

{

    spapr->epow_irq = xics_alloc(spapr->icp, 0, 0, false);

    spapr->epow_notifier.notify = spapr_powerdown_req;

    qemu_register_powerdown_notifier(&spapr->epow_notifier);

    spapr_rtas_register(RTAS_CHECK_EXCEPTION, "check-exception",

                        check_exception);

}
