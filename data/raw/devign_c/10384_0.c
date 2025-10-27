int css_do_rchp(uint8_t cssid, uint8_t chpid)

{

    uint8_t real_cssid;



    if (cssid > channel_subsys.max_cssid) {

        return -EINVAL;

    }

    if (channel_subsys.max_cssid == 0) {

        real_cssid = channel_subsys.default_cssid;

    } else {

        real_cssid = cssid;

    }

    if (!channel_subsys.css[real_cssid]) {

        return -EINVAL;

    }



    if (!channel_subsys.css[real_cssid]->chpids[chpid].in_use) {

        return -ENODEV;

    }



    if (!channel_subsys.css[real_cssid]->chpids[chpid].is_virtual) {

        fprintf(stderr,

                "rchp unsupported for non-virtual chpid %x.%02x!\n",

                real_cssid, chpid);

        return -ENODEV;

    }



    /* We don't really use a channel path, so we're done here. */

    css_queue_crw(CRW_RSC_CHP, CRW_ERC_INIT,

                  channel_subsys.max_cssid > 0 ? 1 : 0, chpid);

    if (channel_subsys.max_cssid > 0) {

        css_queue_crw(CRW_RSC_CHP, CRW_ERC_INIT, 0, real_cssid << 8);

    }

    return 0;

}
