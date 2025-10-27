static void ccw_machine_class_init(ObjectClass *oc, void *data)

{

    MachineClass *mc = MACHINE_CLASS(oc);

    NMIClass *nc = NMI_CLASS(oc);



    mc->name = "s390-ccw-virtio";

    mc->alias = "s390-ccw";

    mc->desc = "VirtIO-ccw based S390 machine";

    mc->init = ccw_init;

    mc->block_default_type = IF_VIRTIO;

    mc->no_cdrom = 1;

    mc->no_floppy = 1;

    mc->no_serial = 1;

    mc->no_parallel = 1;

    mc->no_sdcard = 1;

    mc->use_sclp = 1;

    mc->max_cpus = 255;

    mc->is_default = 1;

    nc->nmi_monitor_handler = s390_nmi;

}
