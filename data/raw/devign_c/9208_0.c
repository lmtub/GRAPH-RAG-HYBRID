uint16_t css_build_subchannel_id(SubchDev *sch)

{

    if (channel_subsys.max_cssid > 0) {

        return (sch->cssid << 8) | (1 << 3) | (sch->ssid << 1) | 1;

    }

    return (sch->ssid << 1) | 1;

}
