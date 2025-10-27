void qdev_prop_set_ptr(DeviceState *dev, const char *name, void *value)

{

    qdev_prop_set(dev, name, &value, PROP_TYPE_PTR);

}
