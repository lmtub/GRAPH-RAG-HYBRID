void bt_l2cap_psm_register(struct bt_l2cap_device_s *dev, int psm, int min_mtu,

                int (*new_channel)(struct bt_l2cap_device_s *dev,

                        struct bt_l2cap_conn_params_s *params))

{

    struct bt_l2cap_psm_s *new_psm = l2cap_psm(dev, psm);



    if (new_psm) {

        fprintf(stderr, "%s: PSM %04x already registered for device `%s'.\n",

                        __FUNCTION__, psm, dev->device.lmp_name);

        exit(-1);

    }



    new_psm = g_malloc0(sizeof(*new_psm));

    new_psm->psm = psm;

    new_psm->min_mtu = min_mtu;

    new_psm->new_channel = new_channel;

    new_psm->next = dev->first_psm;

    dev->first_psm = new_psm;

}
