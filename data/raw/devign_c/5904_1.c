SubchDev *css_create_sch(CssDevId bus_id, bool is_virtual, bool squash_mcss,

                         Error **errp)

{

    uint16_t schid = 0;

    SubchDev *sch;



    if (bus_id.valid) {

        if (is_virtual != (bus_id.cssid == VIRTUAL_CSSID)) {

            error_setg(errp, "cssid %hhx not valid for %s devices",

                       bus_id.cssid,

                       (is_virtual ? "virtual" : "non-virtual"));

            return NULL;

        }

    }



    if (bus_id.valid) {

        if (squash_mcss) {

            bus_id.cssid = channel_subsys.default_cssid;

        } else if (!channel_subsys.css[bus_id.cssid]) {

            css_create_css_image(bus_id.cssid, false);

        }



        if (!css_find_free_subch_for_devno(bus_id.cssid, bus_id.ssid,

                                           bus_id.devid, &schid, errp)) {

            return NULL;

        }

    } else if (squash_mcss || is_virtual) {

        bus_id.cssid = channel_subsys.default_cssid;



        if (!css_find_free_subch_and_devno(bus_id.cssid, &bus_id.ssid,

                                           &bus_id.devid, &schid, errp)) {

            return NULL;

        }

    } else {

        for (bus_id.cssid = 0; bus_id.cssid < MAX_CSSID; ++bus_id.cssid) {

            if (bus_id.cssid == VIRTUAL_CSSID) {

                continue;

            }



            if (!channel_subsys.css[bus_id.cssid]) {

                css_create_css_image(bus_id.cssid, false);

            }



            if   (css_find_free_subch_and_devno(bus_id.cssid, &bus_id.ssid,

                                                &bus_id.devid, &schid,

                                                NULL)) {

                break;

            }

            if (bus_id.cssid == MAX_CSSID) {

                error_setg(errp, "Virtual channel subsystem is full!");

                return NULL;

            }

        }

    }



    sch = g_new0(SubchDev, 1);

    sch->cssid = bus_id.cssid;

    sch->ssid = bus_id.ssid;

    sch->devno = bus_id.devid;

    sch->schid = schid;

    css_subch_assign(sch->cssid, sch->ssid, schid, sch->devno, sch);

    return sch;

}
