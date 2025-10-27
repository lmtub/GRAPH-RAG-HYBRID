void ssi_register_slave(SSISlaveInfo *info)

{

    assert(info->qdev.size >= sizeof(SSISlave));

    info->qdev.init = ssi_slave_init;

    info->qdev.bus_type = BUS_TYPE_SSI;

    qdev_register(&info->qdev);

}
