void bt_device_done(struct bt_device_s *dev)

{

    struct bt_device_s **p = &dev->net->slave;



    while (*p && *p != dev)

        p = &(*p)->next;

    if (*p != dev) {

        fprintf(stderr, "%s: bad bt device \"%s\"\n", __FUNCTION__,

                        dev->lmp_name ?: "(null)");

        exit(-1);

    }



    *p = dev->next;

}
