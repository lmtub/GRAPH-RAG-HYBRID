void subch_device_save(SubchDev *s, QEMUFile *f)

{

    int i;



    qemu_put_byte(f, s->cssid);

    qemu_put_byte(f, s->ssid);

    qemu_put_be16(f, s->schid);

    qemu_put_be16(f, s->devno);

    qemu_put_byte(f, s->thinint_active);

    /* SCHIB */

    /*     PMCW */

    qemu_put_be32(f, s->curr_status.pmcw.intparm);

    qemu_put_be16(f, s->curr_status.pmcw.flags);

    qemu_put_be16(f, s->curr_status.pmcw.devno);

    qemu_put_byte(f, s->curr_status.pmcw.lpm);

    qemu_put_byte(f, s->curr_status.pmcw.pnom);

    qemu_put_byte(f, s->curr_status.pmcw.lpum);

    qemu_put_byte(f, s->curr_status.pmcw.pim);

    qemu_put_be16(f, s->curr_status.pmcw.mbi);

    qemu_put_byte(f, s->curr_status.pmcw.pom);

    qemu_put_byte(f, s->curr_status.pmcw.pam);

    qemu_put_buffer(f, s->curr_status.pmcw.chpid, 8);

    qemu_put_be32(f, s->curr_status.pmcw.chars);

    /*     SCSW */

    qemu_put_be16(f, s->curr_status.scsw.flags);

    qemu_put_be16(f, s->curr_status.scsw.ctrl);

    qemu_put_be32(f, s->curr_status.scsw.cpa);

    qemu_put_byte(f, s->curr_status.scsw.dstat);

    qemu_put_byte(f, s->curr_status.scsw.cstat);

    qemu_put_be16(f, s->curr_status.scsw.count);

    qemu_put_be64(f, s->curr_status.mba);

    qemu_put_buffer(f, s->curr_status.mda, 4);

    /* end SCHIB */

    qemu_put_buffer(f, s->sense_data, 32);

    qemu_put_be64(f, s->channel_prog);

    /* last cmd */

    qemu_put_byte(f, s->last_cmd.cmd_code);

    qemu_put_byte(f, s->last_cmd.flags);

    qemu_put_be16(f, s->last_cmd.count);

    qemu_put_be32(f, s->last_cmd.cda);

    qemu_put_byte(f, s->last_cmd_valid);

    qemu_put_byte(f, s->id.reserved);

    qemu_put_be16(f, s->id.cu_type);

    qemu_put_byte(f, s->id.cu_model);

    qemu_put_be16(f, s->id.dev_type);

    qemu_put_byte(f, s->id.dev_model);

    qemu_put_byte(f, s->id.unused);

    for (i = 0; i < ARRAY_SIZE(s->id.ciw); i++) {

        qemu_put_byte(f, s->id.ciw[i].type);

        qemu_put_byte(f, s->id.ciw[i].command);

        qemu_put_be16(f, s->id.ciw[i].count);

    }

    qemu_put_byte(f, s->ccw_fmt_1);

    qemu_put_byte(f, s->ccw_no_data_cnt);

}
