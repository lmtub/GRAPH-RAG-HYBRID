static SlirpState *slirp_lookup(Monitor *mon, const char *vlan,

                                const char *stack)

{

    VLANClientState *vc;



    if (vlan) {

        vc = qemu_find_vlan_client_by_name(mon, strtol(vlan, NULL, 0), stack);

        if (!vc) {

            return NULL;

        }

        if (strcmp(vc->model, "user")) {

            monitor_printf(mon, "invalid device specified\n");

            return NULL;

        }

        return vc->opaque;

    } else {

        if (TAILQ_EMPTY(&slirp_stacks)) {

            monitor_printf(mon, "user mode network stack not in use\n");

            return NULL;

        }

        return TAILQ_FIRST(&slirp_stacks);

    }

}
