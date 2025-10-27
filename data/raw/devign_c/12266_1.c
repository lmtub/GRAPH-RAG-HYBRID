static int acpi_load_old(QEMUFile *f, void *opaque, int version_id)

{

    PIIX4PMState *s = opaque;

    int ret, i;

    uint16_t temp;



    ret = pci_device_load(&s->dev, f);

    if (ret < 0) {

        return ret;

    }

    qemu_get_be16s(f, &s->ar.pm1.evt.sts);

    qemu_get_be16s(f, &s->ar.pm1.evt.en);

    qemu_get_be16s(f, &s->ar.pm1.cnt.cnt);



    ret = vmstate_load_state(f, &vmstate_apm, opaque, 1);

    if (ret) {

        return ret;

    }



    qemu_get_timer(f, s->ar.tmr.timer);

    qemu_get_sbe64s(f, &s->ar.tmr.overflow_time);



    qemu_get_be16s(f, (uint16_t *)s->ar.gpe.sts);

    for (i = 0; i < 3; i++) {

        qemu_get_be16s(f, &temp);

    }



    qemu_get_be16s(f, (uint16_t *)s->ar.gpe.en);

    for (i = 0; i < 3; i++) {

        qemu_get_be16s(f, &temp);

    }



    ret = vmstate_load_state(f, &vmstate_pci_status, opaque, 1);

    return ret;

}
