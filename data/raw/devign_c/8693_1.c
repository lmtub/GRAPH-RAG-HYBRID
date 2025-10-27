static int net_slirp_init(VLANState *vlan, const char *model, const char *name)

{

    if (!slirp_inited) {

        slirp_inited = 1;

        slirp_init(slirp_restrict, slirp_ip);

    }

    slirp_vc = qemu_new_vlan_client(vlan, model, name,

                                    slirp_receive, NULL, NULL);

    slirp_vc->info_str[0] = '\0';

    return 0;

}
