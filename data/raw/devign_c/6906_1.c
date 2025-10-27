void net_host_device_remove(Monitor *mon, int vlan_id, const char *device)

{

    VLANState *vlan;

    VLANClientState *vc;



    vlan = qemu_find_vlan(vlan_id);

    if (!vlan) {

        monitor_printf(mon, "can't find vlan %d\n", vlan_id);

        return;

    }



   for(vc = vlan->first_client; vc != NULL; vc = vc->next)

        if (!strcmp(vc->name, device))

            break;



    if (!vc) {

        monitor_printf(mon, "can't find device %s\n", device);

        return;

    }

    qemu_del_vlan_client(vc);

}
