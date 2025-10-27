static int vio_make_devnode(VIOsPAPRDevice *dev,

                            void *fdt)

{

    VIOsPAPRDeviceInfo *info = (VIOsPAPRDeviceInfo *)qdev_get_info(&dev->qdev);

    int vdevice_off, node_off, ret;

    char *dt_name;



    vdevice_off = fdt_path_offset(fdt, "/vdevice");

    if (vdevice_off < 0) {

        return vdevice_off;

    }



    dt_name = vio_format_dev_name(dev);

    if (!dt_name) {

        return -ENOMEM;

    }



    node_off = fdt_add_subnode(fdt, vdevice_off, dt_name);

    free(dt_name);

    if (node_off < 0) {

        return node_off;

    }



    ret = fdt_setprop_cell(fdt, node_off, "reg", dev->reg);

    if (ret < 0) {

        return ret;

    }



    if (info->dt_type) {

        ret = fdt_setprop_string(fdt, node_off, "device_type",

                                 info->dt_type);

        if (ret < 0) {

            return ret;

        }

    }



    if (info->dt_compatible) {

        ret = fdt_setprop_string(fdt, node_off, "compatible",

                                 info->dt_compatible);

        if (ret < 0) {

            return ret;

        }

    }



    if (dev->qirq) {

        uint32_t ints_prop[] = {cpu_to_be32(dev->vio_irq_num), 0};



        ret = fdt_setprop(fdt, node_off, "interrupts", ints_prop,

                          sizeof(ints_prop));

        if (ret < 0) {

            return ret;

        }

    }



    if (dev->rtce_window_size) {

        uint32_t dma_prop[] = {cpu_to_be32(dev->reg),

                               0, 0,

                               0, cpu_to_be32(dev->rtce_window_size)};



        ret = fdt_setprop_cell(fdt, node_off, "ibm,#dma-address-cells", 2);

        if (ret < 0) {

            return ret;

        }



        ret = fdt_setprop_cell(fdt, node_off, "ibm,#dma-size-cells", 2);

        if (ret < 0) {

            return ret;

        }



        ret = fdt_setprop(fdt, node_off, "ibm,my-dma-window", dma_prop,

                          sizeof(dma_prop));

        if (ret < 0) {

            return ret;

        }

    }



    if (info->devnode) {

        ret = (info->devnode)(dev, fdt, node_off);

        if (ret < 0) {

            return ret;

        }

    }



    return node_off;

}
