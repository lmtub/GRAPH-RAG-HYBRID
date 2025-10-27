void qdev_prop_set_drive(DeviceState *dev, const char *name,

                         BlockBackend *value, Error **errp)

{

    object_property_set_str(OBJECT(dev), value ? blk_name(value) : "",

                            name, errp);

}
