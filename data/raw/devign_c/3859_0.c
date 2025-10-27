char *get_boot_devices_list(uint32_t *size)

{

    FWBootEntry *i;

    uint32_t total = 0;

    char *list = NULL;



    QTAILQ_FOREACH(i, &fw_boot_order, link) {

        char *devpath = NULL, *bootpath;

        int len;



        if (i->dev) {

            devpath = qdev_get_fw_dev_path(i->dev);

            assert(devpath);

        }



        if (i->suffix && devpath) {

            bootpath = qemu_malloc(strlen(devpath) + strlen(i->suffix) + 1);

            sprintf(bootpath, "%s%s", devpath, i->suffix);

            qemu_free(devpath);

        } else if (devpath) {

            bootpath = devpath;

        } else {

            bootpath = strdup(i->suffix);

            assert(bootpath);

        }



        if (total) {

            list[total-1] = '\n';

        }

        len = strlen(bootpath) + 1;

        list = qemu_realloc(list, total + len);

        memcpy(&list[total], bootpath, len);

        total += len;

        qemu_free(bootpath);

    }



    *size = total;



    return list;

}
