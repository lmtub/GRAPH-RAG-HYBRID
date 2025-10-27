static void sch_handle_start_func(SubchDev *sch, ORB *orb)

{



    PMCW *p = &sch->curr_status.pmcw;

    SCSW *s = &sch->curr_status.scsw;

    int path;

    int ret;



    /* Path management: In our simple css, we always choose the only path. */

    path = 0x80;



    if (!(s->ctrl & SCSW_ACTL_SUSP)) {



        /* Look at the orb and try to execute the channel program. */

        assert(orb != NULL); /* resume does not pass an orb */

        p->intparm = orb->intparm;

        if (!(orb->lpm & path)) {

            /* Generate a deferred cc 3 condition. */

            s->flags |= SCSW_FLAGS_MASK_CC;

            s->ctrl &= ~SCSW_CTRL_MASK_STCTL;

            s->ctrl |= (SCSW_STCTL_ALERT | SCSW_STCTL_STATUS_PEND);

            return;

        }

        sch->ccw_fmt_1 = !!(orb->ctrl0 & ORB_CTRL0_MASK_FMT);

        sch->ccw_no_data_cnt = 0;

    } else {

        s->ctrl &= ~(SCSW_ACTL_SUSP | SCSW_ACTL_RESUME_PEND);

    }

    sch->last_cmd_valid = false;

    do {

        ret = css_interpret_ccw(sch, sch->channel_prog);

        switch (ret) {

        case -EAGAIN:

            /* ccw chain, continue processing */

            break;

        case 0:

            /* success */

            s->ctrl &= ~SCSW_ACTL_START_PEND;

            s->ctrl &= ~SCSW_CTRL_MASK_STCTL;

            s->ctrl |= SCSW_STCTL_PRIMARY | SCSW_STCTL_SECONDARY |

                    SCSW_STCTL_STATUS_PEND;

            s->dstat = SCSW_DSTAT_CHANNEL_END | SCSW_DSTAT_DEVICE_END;

            s->cpa = sch->channel_prog + 8;

            break;

        case -ENOSYS:

            /* unsupported command, generate unit check (command reject) */

            s->ctrl &= ~SCSW_ACTL_START_PEND;

            s->dstat = SCSW_DSTAT_UNIT_CHECK;

            /* Set sense bit 0 in ecw0. */

            sch->sense_data[0] = 0x80;

            s->ctrl &= ~SCSW_CTRL_MASK_STCTL;

            s->ctrl |= SCSW_STCTL_PRIMARY | SCSW_STCTL_SECONDARY |

                    SCSW_STCTL_ALERT | SCSW_STCTL_STATUS_PEND;

            s->cpa = sch->channel_prog + 8;

            break;

        case -EFAULT:

            /* memory problem, generate channel data check */

            s->ctrl &= ~SCSW_ACTL_START_PEND;

            s->cstat = SCSW_CSTAT_DATA_CHECK;

            s->ctrl &= ~SCSW_CTRL_MASK_STCTL;

            s->ctrl |= SCSW_STCTL_PRIMARY | SCSW_STCTL_SECONDARY |

                    SCSW_STCTL_ALERT | SCSW_STCTL_STATUS_PEND;

            s->cpa = sch->channel_prog + 8;

            break;

        case -EBUSY:

            /* subchannel busy, generate deferred cc 1 */

            s->flags &= ~SCSW_FLAGS_MASK_CC;

            s->flags |= (1 << 8);

            s->ctrl &= ~SCSW_CTRL_MASK_STCTL;

            s->ctrl |= SCSW_STCTL_ALERT | SCSW_STCTL_STATUS_PEND;

            break;

        case -EINPROGRESS:

            /* channel program has been suspended */

            s->ctrl &= ~SCSW_ACTL_START_PEND;

            s->ctrl |= SCSW_ACTL_SUSP;

            break;

        default:

            /* error, generate channel program check */

            s->ctrl &= ~SCSW_ACTL_START_PEND;

            s->cstat = SCSW_CSTAT_PROG_CHECK;

            s->ctrl &= ~SCSW_CTRL_MASK_STCTL;

            s->ctrl |= SCSW_STCTL_PRIMARY | SCSW_STCTL_SECONDARY |

                    SCSW_STCTL_ALERT | SCSW_STCTL_STATUS_PEND;

            s->cpa = sch->channel_prog + 8;

            break;

        }

    } while (ret == -EAGAIN);



}