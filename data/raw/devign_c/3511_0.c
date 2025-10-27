const char *qdev_fw_name(DeviceState *dev)

{

    DeviceClass *dc = DEVICE_GET_CLASS(dev);



    if (dc->fw_name) {

        return dc->fw_name;

    } else if (dc->alias) {

        return dc->alias;

    }



    return object_get_typename(OBJECT(dev));

}
