static int bt_parse(const char *opt)

{

    const char *endp, *p;

    int vlan;



    if (strstart(opt, "hci", &endp)) {

        if (!*endp || *endp == ',') {

            if (*endp)

                if (!strstart(endp, ",vlan=", 0))

                    opt = endp + 1;



            return bt_hci_parse(opt);

       }

    } else if (strstart(opt, "vhci", &endp)) {

        if (!*endp || *endp == ',') {

            if (*endp) {

                if (strstart(endp, ",vlan=", &p)) {

                    vlan = strtol(p, (char **) &endp, 0);

                    if (*endp) {

                        fprintf(stderr, "qemu: bad scatternet '%s'\n", p);

                        return 1;

                    }

                } else {

                    fprintf(stderr, "qemu: bad parameter '%s'\n", endp + 1);

                    return 1;

                }

            } else

                vlan = 0;



            bt_vhci_add(vlan);

            return 0;

        }

    } else if (strstart(opt, "device:", &endp))

        return !bt_device_add(endp);



    fprintf(stderr, "qemu: bad bluetooth parameter '%s'\n", opt);

    return 1;

}
