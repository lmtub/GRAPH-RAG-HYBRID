int css_do_ssch(SubchDev *sch, ORB *orb)

{

    SCSW *s = &sch->curr_status.scsw;

    PMCW *p = &sch->curr_status.pmcw;

    int ret;



    if (!(p->flags & (PMCW_FLAGS_MASK_DNV | PMCW_FLAGS_MASK_ENA))) {

        ret = -ENODEV;

        goto out;

    }



    if (s->ctrl & SCSW_STCTL_STATUS_PEND) {

        ret = -EINPROGRESS;

        goto out;

    }



    if (s->ctrl & (SCSW_FCTL_START_FUNC |

                   SCSW_FCTL_HALT_FUNC |

                   SCSW_FCTL_CLEAR_FUNC)) {

        ret = -EBUSY;

        goto out;

    }



    /* If monitoring is active, update counter. */

    if (channel_subsys.chnmon_active) {

        css_update_chnmon(sch);

    }

    sch->channel_prog = orb->cpa;

    /* Trigger the start function. */

    s->ctrl |= (SCSW_FCTL_START_FUNC | SCSW_ACTL_START_PEND);

    s->flags &= ~SCSW_FLAGS_MASK_PNO;



    do_subchannel_work(sch, orb);

    ret = 0;



out:

    return ret;

}
