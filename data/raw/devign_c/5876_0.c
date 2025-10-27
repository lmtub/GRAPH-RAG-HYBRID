int css_do_tsch_get_irb(SubchDev *sch, IRB *target_irb, int *irb_len)

{

    SCSW *s = &sch->curr_status.scsw;

    PMCW *p = &sch->curr_status.pmcw;

    uint16_t stctl;

    IRB irb;



    if (!(p->flags & (PMCW_FLAGS_MASK_DNV | PMCW_FLAGS_MASK_ENA))) {

        return 3;

    }



    stctl = s->ctrl & SCSW_CTRL_MASK_STCTL;



    /* Prepare the irb for the guest. */

    memset(&irb, 0, sizeof(IRB));



    /* Copy scsw from current status. */

    memcpy(&irb.scsw, s, sizeof(SCSW));

    if (stctl & SCSW_STCTL_STATUS_PEND) {

        if (s->cstat & (SCSW_CSTAT_DATA_CHECK |

                        SCSW_CSTAT_CHN_CTRL_CHK |

                        SCSW_CSTAT_INTF_CTRL_CHK)) {

            irb.scsw.flags |= SCSW_FLAGS_MASK_ESWF;

            irb.esw[0] = 0x04804000;

        } else {

            irb.esw[0] = 0x00800000;

        }

        /* If a unit check is pending, copy sense data. */

        if ((s->dstat & SCSW_DSTAT_UNIT_CHECK) &&

            (p->chars & PMCW_CHARS_MASK_CSENSE)) {

            int i;



            irb.scsw.flags |= SCSW_FLAGS_MASK_ESWF | SCSW_FLAGS_MASK_ECTL;

            /* Attention: sense_data is already BE! */

            memcpy(irb.ecw, sch->sense_data, sizeof(sch->sense_data));

            for (i = 0; i < ARRAY_SIZE(irb.ecw); i++) {

                irb.ecw[i] = be32_to_cpu(irb.ecw[i]);

            }

            irb.esw[1] = 0x01000000 | (sizeof(sch->sense_data) << 8);

        }

    }

    /* Store the irb to the guest. */

    copy_irb_to_guest(target_irb, &irb, p, irb_len);



    return ((stctl & SCSW_STCTL_STATUS_PEND) == 0);

}
