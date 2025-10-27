int css_do_hsch(SubchDev *sch)

{

    SCSW *s = &sch->curr_status.scsw;

    PMCW *p = &sch->curr_status.pmcw;

    int ret;



    if (!(p->flags & (PMCW_FLAGS_MASK_DNV | PMCW_FLAGS_MASK_ENA))) {

        ret = -ENODEV;

        goto out;

    }



    if (((s->ctrl & SCSW_CTRL_MASK_STCTL) == SCSW_STCTL_STATUS_PEND) ||

        (s->ctrl & (SCSW_STCTL_PRIMARY |

                    SCSW_STCTL_SECONDARY |

                    SCSW_STCTL_ALERT))) {

        ret = -EINPROGRESS;

        goto out;

    }



    if (s->ctrl & (SCSW_FCTL_HALT_FUNC | SCSW_FCTL_CLEAR_FUNC)) {

        ret = -EBUSY;

        goto out;

    }



    /* Trigger the halt function. */

    s->ctrl |= SCSW_FCTL_HALT_FUNC;

    s->ctrl &= ~SCSW_FCTL_START_FUNC;

    if (((s->ctrl & SCSW_CTRL_MASK_ACTL) ==

         (SCSW_ACTL_SUBCH_ACTIVE | SCSW_ACTL_DEVICE_ACTIVE)) &&

        ((s->ctrl & SCSW_CTRL_MASK_STCTL) == SCSW_STCTL_INTERMEDIATE)) {

        s->ctrl &= ~SCSW_STCTL_STATUS_PEND;

    }

    s->ctrl |= SCSW_ACTL_HALT_PEND;



    do_subchannel_work(sch, NULL);

    ret = 0;



out:

    return ret;

}
