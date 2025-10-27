void spapr_vio_bus_register_withprop(VIOsPAPRDeviceInfo *info)

{

    info->qdev.init = spapr_vio_busdev_init;

    info->qdev.bus_info = &spapr_vio_bus_info;



    assert(info->qdev.size >= sizeof(VIOsPAPRDevice));

    qdev_register(&info->qdev);

}
