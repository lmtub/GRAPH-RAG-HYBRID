void isa_ne2000_init(int base, int irq, NICInfo *nd)

{

    ISADevice *dev;



    qemu_check_nic_model(nd, "ne2k_isa");



    dev = isa_create("ne2k_isa");

    dev->qdev.nd = nd; /* hack alert */

    qdev_prop_set_uint32(&dev->qdev, "iobase", base);

    qdev_prop_set_uint32(&dev->qdev, "irq",    irq);

    qdev_init(&dev->qdev);

}
