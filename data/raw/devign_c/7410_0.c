static void ipmi_sim_realize(DeviceState *dev, Error **errp)

{

    IPMIBmc *b = IPMI_BMC(dev);

    unsigned int i;

    IPMIBmcSim *ibs = IPMI_BMC_SIMULATOR(b);



    qemu_mutex_init(&ibs->lock);

    QTAILQ_INIT(&ibs->rcvbufs);



    ibs->bmc_global_enables = (1 << IPMI_BMC_EVENT_LOG_BIT);

    ibs->device_id = 0x20;

    ibs->ipmi_version = 0x02; /* IPMI 2.0 */

    ibs->restart_cause = 0;

    for (i = 0; i < 4; i++) {

        ibs->sel.last_addition[i] = 0xff;

        ibs->sel.last_clear[i] = 0xff;

        ibs->sdr.last_addition[i] = 0xff;

        ibs->sdr.last_clear[i] = 0xff;

    }



    ipmi_sdr_init(ibs);



    ibs->acpi_power_state[0] = 0;

    ibs->acpi_power_state[1] = 0;



    if (qemu_uuid_set) {

        memcpy(&ibs->uuid, qemu_uuid, 16);

    } else {

        memset(&ibs->uuid, 0, 16);

    }



    ipmi_init_sensors_from_sdrs(ibs);

    register_cmds(ibs);



    ibs->timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, ipmi_timeout, ibs);



    vmstate_register(NULL, 0, &vmstate_ipmi_sim, ibs);

}
