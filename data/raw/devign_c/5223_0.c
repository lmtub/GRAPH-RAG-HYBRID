static struct bt_device_s *bt_device_add(const char *opt)

{

    struct bt_scatternet_s *vlan;

    int vlan_id = 0;

    char *endp = strstr(opt, ",vlan=");

    int len = (endp ? endp - opt : strlen(opt)) + 1;

    char devname[10];



    pstrcpy(devname, MIN(sizeof(devname), len), opt);



    if (endp) {

        vlan_id = strtol(endp + 6, &endp, 0);

        if (*endp) {

            fprintf(stderr, "qemu: unrecognised bluetooth vlan Id\n");

            return 0;

        }

    }



    vlan = qemu_find_bt_vlan(vlan_id);



    if (!vlan->slave)

        fprintf(stderr, "qemu: warning: adding a slave device to "

                        "an empty scatternet %i\n", vlan_id);



    if (!strcmp(devname, "keyboard"))

        return bt_keyboard_init(vlan);



    fprintf(stderr, "qemu: unsupported bluetooth device `%s'\n", devname);

    return 0;

}
