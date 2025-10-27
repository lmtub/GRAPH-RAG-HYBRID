static char *idebus_get_fw_dev_path(DeviceState *dev)

{

    char path[30];



    snprintf(path, sizeof(path), "%s@%d", qdev_fw_name(dev),

             ((IDEBus*)dev->parent_bus)->bus_id);



    return strdup(path);

}
