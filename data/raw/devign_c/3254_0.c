int css_do_csch(SubchDev *sch)

{

    SCSW *s = &sch->curr_status.scsw;

    PMCW *p = &sch->curr_status.pmcw;

    int ret;



    if (!(p->flags & (PMCW_FLAGS_MASK_DNV | PMCW_FLAGS_MASK_ENA))) {

        ret = -ENODEV;

        goto out;

    }



    /* Trigger the clear function. */

    s->ctrl &= ~(SCSW_CTRL_MASK_FCTL | SCSW_CTRL_MASK_ACTL);

    s->ctrl |= SCSW_FCTL_CLEAR_FUNC | SCSW_ACTL_CLEAR_PEND;



    do_subchannel_work(sch, NULL);

    ret = 0;



out:

    return ret;

}
