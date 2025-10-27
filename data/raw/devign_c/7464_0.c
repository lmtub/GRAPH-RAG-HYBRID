int do_subchannel_work_passthrough(SubchDev *sch)

{

    int ret;

    SCSW *s = &sch->curr_status.scsw;



    if (s->ctrl & SCSW_FCTL_CLEAR_FUNC) {

        /* TODO: Clear handling */

        sch_handle_clear_func(sch);

        ret = 0;

    } else if (s->ctrl & SCSW_FCTL_HALT_FUNC) {

        /* TODO: Halt handling */

        sch_handle_halt_func(sch);

        ret = 0;

    } else if (s->ctrl & SCSW_FCTL_START_FUNC) {

        ret = sch_handle_start_func_passthrough(sch);

    } else {

        /* Cannot happen. */

        return -ENODEV;

    }



    return ret;

}
