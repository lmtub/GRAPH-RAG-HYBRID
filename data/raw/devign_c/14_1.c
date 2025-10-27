int net_init_tap(QemuOpts *opts, const char *name, VLANState *vlan)

{

    const char *ifname;



    ifname = qemu_opt_get(opts, "ifname");



    if (!ifname) {

        error_report("tap: no interface name");

        return -1;

    }



    if (tap_win32_init(vlan, "tap", name, ifname) == -1) {

        return -1;

    }



    return 0;

}
