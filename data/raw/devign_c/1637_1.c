static void ccw_init(MachineState *machine)

{

    int ret;

    VirtualCssBus *css_bus;



    s390_sclp_init();

    s390_memory_init(machine->ram_size);



    /* init CPUs (incl. CPU model) early so s390_has_feature() works */

    s390_init_cpus(machine);



    s390_flic_init();



    /* get a BUS */

    css_bus = virtual_css_bus_init();

    s390_init_ipl_dev(machine->kernel_filename, machine->kernel_cmdline,

                      machine->initrd_filename, "s390-ccw.img",

                      "s390-netboot.img", true);



    if (s390_has_feat(S390_FEAT_ZPCI)) {

        DeviceState *dev = qdev_create(NULL, TYPE_S390_PCI_HOST_BRIDGE);

        object_property_add_child(qdev_get_machine(),

                                  TYPE_S390_PCI_HOST_BRIDGE,

                                  OBJECT(dev), NULL);

        qdev_init_nofail(dev);

    }



    /* register hypercalls */

    virtio_ccw_register_hcalls();



    s390_enable_css_support(s390_cpu_addr2state(0));

    /*

     * Non mcss-e enabled guests only see the devices from the default

     * css, which is determined by the value of the squash_mcss property.

     * Note: we must not squash non virtual devices to css 0xFE.

     */

    if (css_bus->squash_mcss) {

        ret = css_create_css_image(0, true);

    } else {

        ret = css_create_css_image(VIRTUAL_CSSID, true);

    }

    assert(ret == 0);



    /* Create VirtIO network adapters */

    s390_create_virtio_net(BUS(css_bus), "virtio-net-ccw");



    /* Register savevm handler for guest TOD clock */

    register_savevm_live(NULL, "todclock", 0, 1, &savevm_gtod, NULL);

}
