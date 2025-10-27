static void copy_irb_to_guest(IRB *dest, const IRB *src, PMCW *pmcw)

{

    int i;

    uint16_t stctl = src->scsw.ctrl & SCSW_CTRL_MASK_STCTL;

    uint16_t actl = src->scsw.ctrl & SCSW_CTRL_MASK_ACTL;



    copy_scsw_to_guest(&dest->scsw, &src->scsw);



    for (i = 0; i < ARRAY_SIZE(dest->esw); i++) {

        dest->esw[i] = cpu_to_be32(src->esw[i]);

    }

    for (i = 0; i < ARRAY_SIZE(dest->ecw); i++) {

        dest->ecw[i] = cpu_to_be32(src->ecw[i]);

    }

    /* extended measurements enabled? */

    if ((src->scsw.flags & SCSW_FLAGS_MASK_ESWF) ||

        !(pmcw->flags & PMCW_FLAGS_MASK_TF) ||

        !(pmcw->chars & PMCW_CHARS_MASK_XMWME)) {

        return;

    }

    /* extended measurements pending? */

    if (!(stctl & SCSW_STCTL_STATUS_PEND)) {

        return;

    }

    if ((stctl & SCSW_STCTL_PRIMARY) ||

        (stctl == SCSW_STCTL_SECONDARY) ||

        ((stctl & SCSW_STCTL_INTERMEDIATE) && (actl & SCSW_ACTL_SUSP))) {

        for (i = 0; i < ARRAY_SIZE(dest->emw); i++) {

            dest->emw[i] = cpu_to_be32(src->emw[i]);

        }

    }

}
