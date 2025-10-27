static USBDevice *usb_net_init(const char *cmdline)

{

    USBDevice *dev;

    QemuOpts *opts;

    int idx;



    opts = qemu_opts_parse(&qemu_net_opts, cmdline, NULL);

    if (!opts) {



    qemu_opt_set(opts, "type", "nic");

    qemu_opt_set(opts, "model", "usb");



    idx = net_client_init(NULL, opts, 0);

    if (idx == -1) {





    dev = usb_create(NULL /* FIXME */, "usb-net");




    qdev_set_nic_properties(&dev->qdev, &nd_table[idx]);

    qdev_init_nofail(&dev->qdev);

    return dev;
