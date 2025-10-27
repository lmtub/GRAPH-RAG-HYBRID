static int nic_load(QEMUFile * f, void *opaque, int version_id)

{

    EEPRO100State *s = opaque;

    int i;

    int ret;



    if (version_id > 3)

        return -EINVAL;



    if (version_id >= 3) {

        ret = pci_device_load(&s->dev, f);

        if (ret < 0)

            return ret;

    }



    if (version_id >= 2) {

        qemu_get_8s(f, &s->rxcr);

    } else {

        s->rxcr = 0x0c;

    }



    qemu_get_8s(f, &s->cmd);

    qemu_get_be32s(f, &s->start);

    qemu_get_be32s(f, &s->stop);

    qemu_get_8s(f, &s->boundary);

    qemu_get_8s(f, &s->tsr);

    qemu_get_8s(f, &s->tpsr);

    qemu_get_be16s(f, &s->tcnt);

    qemu_get_be16s(f, &s->rcnt);

    qemu_get_be32s(f, &s->rsar);

    qemu_get_8s(f, &s->rsr);

    qemu_get_8s(f, &s->isr);

    qemu_get_8s(f, &s->dcfg);

    qemu_get_8s(f, &s->imr);

    qemu_get_buffer(f, s->phys, 6);

    qemu_get_8s(f, &s->curpag);

    qemu_get_buffer(f, s->mult, 8);

    qemu_get_buffer(f, s->mem, sizeof(s->mem));



    /* Restore all members of struct between scv_stat and mem. */

    qemu_get_8s(f, &s->scb_stat);

    qemu_get_8s(f, &s->int_stat);

    for (i = 0; i < 3; i++) {

        qemu_get_be32s(f, &s->region[i]);

    }

    qemu_get_buffer(f, s->macaddr, 6);

    for (i = 0; i < 19; i++) {

        qemu_get_be32s(f, &s->statcounter[i]);

    }

    for (i = 0; i < 32; i++) {

        qemu_get_be16s(f, &s->mdimem[i]);

    }

    /* The eeprom should be saved and restored by its own routines. */

    qemu_get_be32s(f, &s->device);

    qemu_get_be32s(f, &s->pointer);

    qemu_get_be32s(f, &s->cu_base);

    qemu_get_be32s(f, &s->cu_offset);

    qemu_get_be32s(f, &s->ru_base);

    qemu_get_be32s(f, &s->ru_offset);

    qemu_get_be32s(f, &s->statsaddr);

    /* Restore epro100_stats_t statistics. */

    qemu_get_be32s(f, &s->statistics.tx_good_frames);

    qemu_get_be32s(f, &s->statistics.tx_max_collisions);

    qemu_get_be32s(f, &s->statistics.tx_late_collisions);

    qemu_get_be32s(f, &s->statistics.tx_underruns);

    qemu_get_be32s(f, &s->statistics.tx_lost_crs);

    qemu_get_be32s(f, &s->statistics.tx_deferred);

    qemu_get_be32s(f, &s->statistics.tx_single_collisions);

    qemu_get_be32s(f, &s->statistics.tx_multiple_collisions);

    qemu_get_be32s(f, &s->statistics.tx_total_collisions);

    qemu_get_be32s(f, &s->statistics.rx_good_frames);

    qemu_get_be32s(f, &s->statistics.rx_crc_errors);

    qemu_get_be32s(f, &s->statistics.rx_alignment_errors);

    qemu_get_be32s(f, &s->statistics.rx_resource_errors);

    qemu_get_be32s(f, &s->statistics.rx_overrun_errors);

    qemu_get_be32s(f, &s->statistics.rx_cdt_errors);

    qemu_get_be32s(f, &s->statistics.rx_short_frame_errors);

    qemu_get_be32s(f, &s->statistics.fc_xmt_pause);

    qemu_get_be32s(f, &s->statistics.fc_rcv_pause);

    qemu_get_be32s(f, &s->statistics.fc_rcv_unsupported);

    qemu_get_be16s(f, &s->statistics.xmt_tco_frames);

    qemu_get_be16s(f, &s->statistics.rcv_tco_frames);

    qemu_get_be32s(f, &s->statistics.complete);

#if 0

    qemu_get_be16s(f, &s->status);

#endif



    /* Configuration bytes. */

    qemu_get_buffer(f, s->configuration, sizeof(s->configuration));



    return 0;

}
